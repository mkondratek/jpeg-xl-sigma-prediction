// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_ANS_H_
#define LIB_JXL_ENC_ANS_H_

// Library to encode the ANS population counts to the bit-stream and encode
// symbols based on the respective distributions.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "lib/jxl/ans_common.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/huffman_table.h"

namespace jxl {

#define USE_MULT_BY_RECIPROCAL

// precision must be equal to:  #bits(state_) + #bits(freq)
#define RECIPROCAL_PRECISION (32 + ANS_LOG_TAB_SIZE)

// Data structure representing one element of the encoding table built
// from a distribution.
// TODO(veluca): split this up, or use an union.
struct ANSEncSymbolInfo {
  // ANS
  uint16_t freq_;
  std::vector<uint16_t> reverse_map_;
#ifdef USE_MULT_BY_RECIPROCAL
  uint64_t ifreq_;
#endif
  // Prefix coding.
  uint8_t depth;
  uint16_t bits;
};

class ANSCoder {
 public:
  ANSCoder() : state_(ANS_SIGNATURE << 16) {}

  uint32_t PutSymbol(const ANSEncSymbolInfo& t, uint8_t* nbits) {
    uint32_t bits = 0;
    *nbits = 0;
    if ((state_ >> (32 - ANS_LOG_TAB_SIZE)) >= t.freq_) {
      bits = state_ & 0xffff;
      state_ >>= 16;
      *nbits = 16;
    }
#ifdef USE_MULT_BY_RECIPROCAL
    // We use mult-by-reciprocal trick, but that requires 64b calc.
    const uint32_t v = (state_ * t.ifreq_) >> RECIPROCAL_PRECISION;
    const uint32_t offset = t.reverse_map_[state_ - v * t.freq_];
    state_ = (v << ANS_LOG_TAB_SIZE) + offset;
#else
    state_ = ((state_ / t.freq_) << ANS_LOG_TAB_SIZE) +
             t.reverse_map_[state_ % t.freq_];
#endif
    return bits;
  }

  uint32_t GetState() const { return state_; }

 private:
  uint32_t state_;
};

// RebalanceHistogram requires a signed type.
using ANSHistBin = int32_t;

struct EntropyEncodingData {
  std::vector<std::vector<ANSEncSymbolInfo>> encoding_info;
  bool use_prefix_code;
  std::vector<HybridUintConfig> uint_config;
  LZ77Params lz77;
};

// Integer to be encoded by an entropy coder, either ANS or Huffman.
struct Token {
  Token(uint32_t c, uint32_t value, uint32_t sigma = -1)
      : is_lz77_length(false), context(c), value(value), sigma(sigma) {}
  uint32_t is_lz77_length : 1;
  uint32_t context : 31;
  uint32_t value;
  uint32_t sigma;
};

// Returns an estimate of the number of bits required to encode the given
// histogram (header bits plus data bits).
float ANSPopulationCost(const ANSHistBin* data, size_t alphabet_size);

// Apply context clustering, compute histograms and encode them. Returns an
// estimate of the total bits used for encoding the stream. If `writer` ==
// nullptr, the bit estimate will not take into account the context map (which
// does not get written if `num_contexts` == 1).
size_t BuildAndEncodeHistograms(const HistogramParams& params,
                                size_t num_contexts,
                                std::vector<std::vector<Token>>& tokens,
                                EntropyEncodingData* codes,
                                std::vector<uint8_t>* context_map,
                                BitWriter* writer, size_t layer,
                                AuxOut* aux_out);

// Write the tokens to a string.
void WriteTokens(const std::vector<Token>& tokens,
                 const EntropyEncodingData& codes,
                 const std::vector<uint8_t>& context_map, BitWriter* writer,
                 size_t layer, AuxOut* aux_out);

// Same as above, but assumes allotment created by caller.
size_t WriteTokens(const std::vector<Token>& tokens,
                   const EntropyEncodingData& codes,
                   const std::vector<uint8_t>& context_map, BitWriter* writer);

// Exposed for tests; to be used with Writer=BitWriter only.
template <typename Writer>
void EncodeUintConfigs(const std::vector<HybridUintConfig>& uint_config,
                       Writer* writer, size_t log_alpha_size);
extern template void EncodeUintConfigs(const std::vector<HybridUintConfig>&,
                                       BitWriter*, size_t);

// Globally set the option to create fuzzer-friendly ANS streams. Negatively
// impacts compression. Not thread-safe.
void SetANSFuzzerFriendly(bool ans_fuzzer_friendly);

////////////////////////////// ANS LAPLACE TABLES ////////////////////////////

template <uint32_t SIGMA_COUNT = 16u, uint32_t MAX_SYMBOL = 256u>
class ANSLaplaceTable {
 public:
  ANSLaplaceTable(uint32_t low, uint32_t high) { CreateTables(low, high); }

  ANSEncSymbolInfo const& get_symbol(uint32_t value, uint32_t sigma) const {
    return m_data[sigma % SIGMA_COUNT][value % MAX_SYMBOL];
  }

  ANSEncSymbolInfo const& get_symbol(Token const& t) const {
    return get_symbol(t.value, t.sigma);
  }

  std::array<std::array<ANSEncSymbolInfo, MAX_SYMBOL>, SIGMA_COUNT> const&
  data() const {
    return m_data;
  }

 private:
  std::array<std::array<ANSHistBin, MAX_SYMBOL>, SIGMA_COUNT> distribution;
  std::array<std::array<ANSEncSymbolInfo, MAX_SYMBOL>, SIGMA_COUNT> m_data = {};

  void CreateTables(uint32_t low, uint32_t high) {
    JXL_ASSERT((high - low) == SIGMA_COUNT);
    for (uint32_t sig = low; sig < high; sig++) {
      CalculateDistribution(sig);
      CreateFreqTable( sig);
    }
  }

  double cdf(double x) {
    double sgn = (x > 0) ? 1 : -1;
    if (std::fpclassify(x) == FP_ZERO) {
      sgn = 0;
    }

    return 0.5 * (sgn * (1 - std::exp(-std::abs(x))) + 1);
  }

  void CalculateDistribution(uint32_t sig) {
    double sigma = 0.125 + 2 * 0.25 * sig;

    for (uint32_t i = 0; i < MAX_SYMBOL; i++) {
      double v = static_cast<double>(static_cast<int>(i) -
                                     static_cast<int>(MAX_SYMBOL / 2));

      if (i == 0) {
        distribution[sig][i] = std::round(1 + ANS_TAB_SIZE * cdf((v + 0.5) / sig));
      } else if (i == MAX_SYMBOL - 1) {
        distribution[sig][i] = std::round(1 + ANS_TAB_SIZE * (1 - cdf((v - 0.5) / sig)));
      } else {
        distribution[sig][i] =
            std::round(1 + ANS_TAB_SIZE * (cdf((v + 0.5) / sigma) - cdf((v - 0.5) / sigma)));
      }
    }

    int sum = std::accumulate(distribution[sig].begin(), distribution[sig].end(), 0);
    ANSHistBin* maxElement = std::max_element(distribution[sig].begin(), distribution[sig].end());
    *maxElement += (ANS_TAB_SIZE - sum);
    while (*maxElement < *(maxElement - 1)) {
      int shift = 1;
      while (*(maxElement - shift) != 1) {
        *maxElement += 2;
        *(maxElement + shift) -= 1;
        *(maxElement - shift) -= 1;
        shift++;
      }
    }
  }

  void CreateFreqTable( uint32_t sig) {
    std::array<ANSEncSymbolInfo, MAX_SYMBOL>& info = m_data[sig];
    const std::array<ANSHistBin, MAX_SYMBOL>& freq = distribution[sig];

    for (uint32_t s = 0u; s < MAX_SYMBOL; s++) {
      info[s].freq_ = static_cast<uint16_t>(freq[s]);
#ifdef USE_MULT_BY_RECIPROCAL
      if (freq[s] != 0) {
        info[s].ifreq_ =
            ((1ull << RECIPROCAL_PRECISION) + info[s].freq_ - 1) / info[s].freq_;
      } else {
        info[s].ifreq_ = 1;  // shouldn't matter (symbol shouldn't occur), but...
      }
#endif
      info[s].reverse_map_.resize(freq[s]);
    }

    // reverse map creation
    Properties counts;
    counts.resize(MAX_SYMBOL);
    for (uint32_t i = 0u; i < MAX_SYMBOL; i++) {
      counts[i] = static_cast<int>(freq[i]);
    }
    size_t log_alpha_size = 8;
    size_t log_entry_size = ANS_LOG_TAB_SIZE - log_alpha_size;
    size_t entry_size_minus_1 = (1 << log_entry_size) - 1;

    AliasTable::Entry a[ANS_TAB_SIZE];
    InitAliasTable(counts, ANS_TAB_SIZE, log_alpha_size, a);
    for (uint32_t i = 0u; i < ANS_TAB_SIZE; i++) {
      AliasTable::Symbol s =
          AliasTable::Lookup(a, i, log_entry_size, entry_size_minus_1);
      info[s.value].reverse_map_[s.offset] = i;
    }
  }
};

////////////////////////////////////////////////////////////////////////////

}  // namespace jxl

#endif  // LIB_JXL_ENC_ANS_H_
