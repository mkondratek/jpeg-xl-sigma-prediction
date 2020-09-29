// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef JXL_DCT_SLOW_H_
#define JXL_DCT_SLOW_H_

// Unoptimized DCT only for use in tests.

#include <string.h>  // memcpy

#include <cmath>
#include <vector>

#include "jxl/common.h"  // Pi

namespace jxl {

namespace test {
static inline double alpha(int u) { return u == 0 ? 0.7071067811865475 : 1.0; }

// N-DCT on M columns.
template <size_t N, size_t M>
void DCT1D(double block[N * M], double out[N * M]) {
  std::vector<double> matrix(N * N);
  const double scale = std::sqrt(2.0 / N);
  for (size_t y = 0; y < N; y++) {
    for (size_t u = 0; u < N; u++) {
      matrix[N * u + y] = alpha(u) * cos((y + 0.5) * u * Pi(1.0 / N)) * scale;
    }
  }
  for (size_t x = 0; x < M; x++) {
    for (size_t u = 0; u < N; u++) {
      out[M * u + x] = 0;
      for (size_t y = 0; y < N; y++) {
        out[M * u + x] += matrix[N * u + y] * block[M * y + x];
      }
    }
  }
}

// N-IDCT on M columns.
template <size_t N, size_t M>
void IDCT1D(double block[N * M], double out[N * M]) {
  std::vector<double> matrix(N * N);
  const double scale = std::sqrt(2.0 / N);
  for (size_t y = 0; y < N; y++) {
    for (size_t u = 0; u < N; u++) {
      // Transpose of DCT matrix.
      matrix[N * y + u] = alpha(u) * cos((y + 0.5) * u * Pi(1.0 / N)) * scale;
    }
  }
  for (size_t x = 0; x < M; x++) {
    for (size_t u = 0; u < N; u++) {
      out[M * u + x] = 0;
      for (size_t y = 0; y < N; y++) {
        out[M * u + x] += matrix[N * u + y] * block[M * y + x];
      }
    }
  }
}

template <size_t N, size_t M>
void TransposeBlock(double in[N * M], double out[M * N]) {
  for (size_t x = 0; x < N; x++) {
    for (size_t y = 0; y < M; y++) {
      out[y * N + x] = in[x * M + y];
    }
  }
}
}  // namespace test

// Untransposed DCT.
template <size_t N>
void DCTSlow(double block[N * N]) {
  constexpr size_t kBlockSize = N * N;
  std::vector<double> g(kBlockSize);
  test::DCT1D<N, N>(block, g.data());
  test::TransposeBlock<N, N>(g.data(), block);
  test::DCT1D<N, N>(block, g.data());
  test::TransposeBlock<N, N>(g.data(), block);
}

// Untransposed IDCT.
template <size_t N>
void IDCTSlow(double block[N * N]) {
  constexpr size_t kBlockSize = N * N;
  std::vector<double> g(kBlockSize);
  test::IDCT1D<N, N>(block, g.data());
  test::TransposeBlock<N, N>(g.data(), block);
  test::IDCT1D<N, N>(block, g.data());
  test::TransposeBlock<N, N>(g.data(), block);
}

}  // namespace jxl

#endif  // JXL_DCT_SLOW_H_
