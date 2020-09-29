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

#include "jxl/color_encoding.h"

#include <array>
#include <cmath>

#include "jxl/color_management.h"
#include "jxl/entropy_coder.h"
#include "jxl/fields.h"

namespace jxl {
namespace {

// These strings are baked into Description - do not change.

std::string ToString(ColorSpace color_space) {
  switch (color_space) {
    case ColorSpace::kRGB:
      return "RGB";
    case ColorSpace::kGray:
      return "Gra";
    case ColorSpace::kXYB:
      return "XYB";
    case ColorSpace::kUnknown:
      return "CS?";
  }
  // Should not happen - visitor fails if enum is invalid.
  JXL_ABORT("Invalid ColorSpace %u", static_cast<uint32_t>(color_space));
}

std::string ToString(WhitePoint white_point) {
  switch (white_point) {
    case WhitePoint::kD65:
      return "D65";
    case WhitePoint::kCustom:
      return "Cst";
    case WhitePoint::kE:
      return "EER";
    case WhitePoint::kDCI:
      return "DCI";
  }
  // Should not happen - visitor fails if enum is invalid.
  JXL_ABORT("Invalid WhitePoint %u", static_cast<uint32_t>(white_point));
}

std::string ToString(Primaries primaries) {
  switch (primaries) {
    case Primaries::kSRGB:
      return "SRG";
    case Primaries::k2100:
      return "202";
    case Primaries::kP3:
      return "DCI";
    case Primaries::kCustom:
      return "Cst";
  }
  // Should not happen - visitor fails if enum is invalid.
  JXL_ABORT("Invalid Primaries %u", static_cast<uint32_t>(primaries));
}

std::string ToString(TransferFunction transfer_function) {
  switch (transfer_function) {
    case TransferFunction::kSRGB:
      return "SRG";
    case TransferFunction::kLinear:
      return "Lin";
    case TransferFunction::k709:
      return "709";
    case TransferFunction::kPQ:
      return "PeQ";
    case TransferFunction::kHLG:
      return "HLG";
    case TransferFunction::kDCI:
      return "DCI";
    case TransferFunction::kUnknown:
      return "TF?";
  }
  // Should not happen - visitor fails if enum is invalid.
  JXL_ABORT("Invalid TransferFunction %u",
            static_cast<uint32_t>(transfer_function));
}

std::string ToString(RenderingIntent rendering_intent) {
  switch (rendering_intent) {
    case RenderingIntent::kPerceptual:
      return "Per";
    case RenderingIntent::kRelative:
      return "Rel";
    case RenderingIntent::kSaturation:
      return "Sat";
    case RenderingIntent::kAbsolute:
      return "Abs";
  }
  // Should not happen - visitor fails if enum is invalid.
  JXL_ABORT("Invalid RenderingIntent %u",
            static_cast<uint32_t>(rendering_intent));
}

template <typename Enum>
Status ParseEnum(const std::string& token, Enum* value) {
  std::string str;
  for (Enum e : Values<Enum>()) {
    if (ToString(e) == token) {
      *value = e;
      return true;
    }
  }
  return false;
}

class Tokenizer {
 public:
  Tokenizer(const std::string* input, char separator)
      : input_(input), separator_(separator) {}

  Status Next(std::string* JXL_RESTRICT next) {
    const size_t end = input_->find(separator_, start_);
    if (end == std::string::npos) {
      *next = input_->substr(start_);  // rest of string
    } else {
      *next = input_->substr(start_, end - start_);
    }
    if (next->empty()) return JXL_FAILURE("Missing token");
    start_ = end + 1;
    return true;
  }

 private:
  const std::string* const input_;  // not owned
  const char separator_;
  size_t start_ = 0;  // of next token
};

Status ParseDouble(const std::string& num, double* JXL_RESTRICT d) {
  char* end;
  errno = 0;
  *d = strtod(num.c_str(), &end);
  if (*d == 0.0 && end == num.c_str()) {
    return JXL_FAILURE("Invalid double: %s", num.c_str());
  }
  if (errno == ERANGE) {
    return JXL_FAILURE("Double out of range: %s", num.c_str());
  }
  return true;
}

Status ParseDouble(Tokenizer* tokenizer, double* JXL_RESTRICT d) {
  std::string num;
  JXL_RETURN_IF_ERROR(tokenizer->Next(&num));
  return ParseDouble(num, d);
}

Status ParseColorSpace(Tokenizer* JXL_RESTRICT tokenizer,
                       ColorEncoding* JXL_RESTRICT c) {
  std::string str;
  JXL_RETURN_IF_ERROR(tokenizer->Next(&str));
  ColorSpace cs;
  if (ParseEnum(str, &cs)) {
    c->SetColorSpace(cs);
    return true;
  }

  return JXL_FAILURE("Unknown ColorSpace %s", str.c_str());
}

Status ParseWhitePoint(Tokenizer* JXL_RESTRICT tokenizer,
                       ColorEncoding* JXL_RESTRICT c) {
  if (c->ImplicitWhitePoint()) return true;

  std::string str;
  JXL_RETURN_IF_ERROR(tokenizer->Next(&str));
  if (ParseEnum(str, &c->white_point)) return true;

  CIExy xy;
  Tokenizer xy_tokenizer(&str, ';');
  JXL_RETURN_IF_ERROR(ParseDouble(&xy_tokenizer, &xy.x));
  JXL_RETURN_IF_ERROR(ParseDouble(&xy_tokenizer, &xy.y));
  if (c->SetWhitePoint(xy)) return true;

  return JXL_FAILURE("Invalid white point %s", str.c_str());
}

Status ParsePrimaries(Tokenizer* JXL_RESTRICT tokenizer,
                      ColorEncoding* JXL_RESTRICT c) {
  if (!c->HasPrimaries()) return true;

  std::string str;
  JXL_RETURN_IF_ERROR(tokenizer->Next(&str));
  if (ParseEnum(str, &c->primaries)) return true;

  PrimariesCIExy xy;
  Tokenizer xy_tokenizer(&str, ';');
  JXL_RETURN_IF_ERROR(ParseDouble(&xy_tokenizer, &xy.r.x));
  JXL_RETURN_IF_ERROR(ParseDouble(&xy_tokenizer, &xy.r.y));
  JXL_RETURN_IF_ERROR(ParseDouble(&xy_tokenizer, &xy.g.x));
  JXL_RETURN_IF_ERROR(ParseDouble(&xy_tokenizer, &xy.g.y));
  JXL_RETURN_IF_ERROR(ParseDouble(&xy_tokenizer, &xy.b.x));
  JXL_RETURN_IF_ERROR(ParseDouble(&xy_tokenizer, &xy.b.y));
  if (c->SetPrimaries(xy)) return true;

  return JXL_FAILURE("Invalid primaries %s", str.c_str());
}

Status ParseRenderingIntent(Tokenizer* JXL_RESTRICT tokenizer,
                            ColorEncoding* JXL_RESTRICT c) {
  std::string str;
  JXL_RETURN_IF_ERROR(tokenizer->Next(&str));
  if (ParseEnum(str, &c->rendering_intent)) return true;

  return JXL_FAILURE("Invalid RenderingIntent %s\n", str.c_str());
}

Status ParseTransferFunction(Tokenizer* JXL_RESTRICT tokenizer,
                             ColorEncoding* JXL_RESTRICT c) {
  if (c->tf.SetImplicit()) return true;

  std::string str;
  JXL_RETURN_IF_ERROR(tokenizer->Next(&str));
  TransferFunction transfer_function;
  if (ParseEnum(str, &transfer_function)) {
    c->tf.SetTransferFunction(transfer_function);
    return true;
  }

  if (str[0] == 'g') {
    double gamma;
    JXL_RETURN_IF_ERROR(ParseDouble(str.substr(1), &gamma));
    if (c->tf.SetGamma(gamma)) return true;
  }

  return JXL_FAILURE("Invalid gamma %s", str.c_str());
}

static double F64FromCustomxyI32(const int32_t i) { return i * 1E-6; }
static Status F64ToCustomxyI32(const double f, int32_t* JXL_RESTRICT i) {
  if (!(-4 <= f && f <= 4)) {
    return JXL_FAILURE("F64 out of bounds for CustomxyI32");
  }
  *i = static_cast<int32_t>(std::round(f * 1E6));
  return true;
}

}  // namespace

CIExy Customxy::Get() const {
  CIExy xy;
  xy.x = F64FromCustomxyI32(x);
  xy.y = F64FromCustomxyI32(y);
  return xy;
}

Status Customxy::Set(const CIExy& xy) {
  JXL_RETURN_IF_ERROR(F64ToCustomxyI32(xy.x, &x));
  JXL_RETURN_IF_ERROR(F64ToCustomxyI32(xy.y, &y));
  size_t extension_bits, total_bits;
  if (!Bundle::CanEncode(*this, &extension_bits, &total_bits)) {
    return JXL_FAILURE("Unable to encode XY %f %f", xy.x, xy.y);
  }
  return true;
}

bool CustomTransferFunction::SetImplicit() {
  if (nonserialized_color_space == ColorSpace::kXYB) {
    if (!SetGamma(1.0 / 3)) JXL_ASSERT(false);
    return true;
  }
  return false;
}

Status CustomTransferFunction::SetGamma(double gamma) {
  if (gamma <= 0.0 || gamma > 1.0) {
    return JXL_FAILURE("Invalid gamma %f", gamma);
  }

  have_gamma_ = false;
  if (ApproxEq(gamma, 1.0)) {
    transfer_function_ = TransferFunction::kLinear;
    return true;
  }
  if (ApproxEq(gamma, 1.0 / 2.6)) {
    transfer_function_ = TransferFunction::kDCI;
    return true;
  }
  // Don't translate 0.45.. to kSRGB nor k709 - that might change pixel
  // values because those curves also have a linear part.

  have_gamma_ = true;
  gamma_ = std::round(gamma * kGammaMul);
  transfer_function_ = TransferFunction::kUnknown;
  return true;
}

namespace {

std::array<ColorEncoding, 2> CreateC2(const Primaries pr,
                                      const TransferFunction tf) {
  std::array<ColorEncoding, 2> c2;

  {
    ColorEncoding* c_rgb = c2.data() + 0;
    c_rgb->SetColorSpace(ColorSpace::kRGB);
    c_rgb->white_point = WhitePoint::kD65;
    c_rgb->primaries = pr;
    c_rgb->tf.SetTransferFunction(tf);
    JXL_CHECK(c_rgb->CreateICC());
  }

  {
    ColorEncoding* c_gray = c2.data() + 1;
    c_gray->SetColorSpace(ColorSpace::kGray);
    c_gray->white_point = WhitePoint::kD65;
    c_gray->primaries = pr;
    c_gray->tf.SetTransferFunction(tf);
    JXL_CHECK(c_gray->CreateICC());
  }

  return c2;
}

}  // namespace

const ColorEncoding& ColorEncoding::SRGB(bool is_gray) {
  static std::array<ColorEncoding, 2> c2 =
      CreateC2(Primaries::kSRGB, TransferFunction::kSRGB);
  return c2[is_gray];
}
const ColorEncoding& ColorEncoding::LinearSRGB(bool is_gray) {
  static std::array<ColorEncoding, 2> c2 =
      CreateC2(Primaries::kSRGB, TransferFunction::kLinear);
  return c2[is_gray];
}

CIExy ColorEncoding::GetWhitePoint() const {
  CIExy xy;
  switch (white_point) {
    case WhitePoint::kCustom:
      return white_.Get();

    case WhitePoint::kD65:
      xy.x = 0.3127;
      xy.y = 0.3290;
      return xy;

    case WhitePoint::kDCI:
      // From https://ieeexplore.ieee.org/document/7290729 C.2 page 11
      xy.x = 0.314;
      xy.y = 0.351;
      return xy;

    case WhitePoint::kE:
      xy.x = xy.y = 1.0 / 3;
      return xy;
  }
  JXL_ABORT("Invalid WhitePoint %u", static_cast<uint32_t>(white_point));
}

Status ColorEncoding::SetWhitePoint(const CIExy& xy) {
  if (xy.x == 0.0 || xy.y == 0.0) {
    return JXL_FAILURE("Invalid white point %f %f", xy.x, xy.y);
  }
  if (ApproxEq(xy.x, 0.3127) && ApproxEq(xy.y, 0.3290)) {
    white_point = WhitePoint::kD65;
    return true;
  }
  if (ApproxEq(xy.x, 1.0 / 3) && ApproxEq(xy.y, 1.0 / 3)) {
    white_point = WhitePoint::kE;
    return true;
  }
  if (ApproxEq(xy.x, 0.314) && ApproxEq(xy.y, 0.351)) {
    white_point = WhitePoint::kDCI;
    return true;
  }
  white_point = WhitePoint::kCustom;
  return white_.Set(xy);
}

PrimariesCIExy ColorEncoding::GetPrimaries() const {
  JXL_ASSERT(HasPrimaries());
  PrimariesCIExy xy;
  switch (primaries) {
    case Primaries::kCustom:
      xy.r = red_.Get();
      xy.g = green_.Get();
      xy.b = blue_.Get();
      return xy;

    case Primaries::kSRGB:
      xy.r.x = 0.639998686;
      xy.r.y = 0.330010138;
      xy.g.x = 0.300003784;
      xy.g.y = 0.600003357;
      xy.b.x = 0.150002046;
      xy.b.y = 0.059997204;
      return xy;

    case Primaries::k2100:
      xy.r.x = 0.708;
      xy.r.y = 0.292;
      xy.g.x = 0.170;
      xy.g.y = 0.797;
      xy.b.x = 0.131;
      xy.b.y = 0.046;
      return xy;

    case Primaries::kP3:
      xy.r.x = 0.680;
      xy.r.y = 0.320;
      xy.g.x = 0.265;
      xy.g.y = 0.690;
      xy.b.x = 0.150;
      xy.b.y = 0.060;
      return xy;
  }
  JXL_ABORT("Invalid Primaries %u", static_cast<uint32_t>(primaries));
}

Status ColorEncoding::SetPrimaries(const PrimariesCIExy& xy) {
  JXL_ASSERT(HasPrimaries());
  if (xy.r.x == 0.0 || xy.r.y == 0.0 || xy.g.x == 0.0 || xy.g.y == 0.0 ||
      xy.b.x == 0.0 || xy.b.y == 0.0) {
    return JXL_FAILURE("Invalid primaries %f %f %f %f %f %f", xy.r.x, xy.r.y,
                       xy.g.x, xy.g.y, xy.b.x, xy.b.y);
  }

  if (ApproxEq(xy.r.x, 0.64) && ApproxEq(xy.r.y, 0.33) &&
      ApproxEq(xy.g.x, 0.30) && ApproxEq(xy.g.y, 0.60) &&
      ApproxEq(xy.b.x, 0.15) && ApproxEq(xy.b.y, 0.06)) {
    primaries = Primaries::kSRGB;
    return true;
  }

  if (ApproxEq(xy.r.x, 0.708) && ApproxEq(xy.r.y, 0.292) &&
      ApproxEq(xy.g.x, 0.170) && ApproxEq(xy.g.y, 0.797) &&
      ApproxEq(xy.b.x, 0.131) && ApproxEq(xy.b.y, 0.046)) {
    primaries = Primaries::k2100;
    return true;
  }
  if (ApproxEq(xy.r.x, 0.680) && ApproxEq(xy.r.y, 0.320) &&
      ApproxEq(xy.g.x, 0.265) && ApproxEq(xy.g.y, 0.690) &&
      ApproxEq(xy.b.x, 0.150) && ApproxEq(xy.b.y, 0.060)) {
    primaries = Primaries::kP3;
    return true;
  }

  primaries = Primaries::kCustom;
  JXL_RETURN_IF_ERROR(red_.Set(xy.r));
  JXL_RETURN_IF_ERROR(green_.Set(xy.g));
  JXL_RETURN_IF_ERROR(blue_.Set(xy.b));
  return true;
}

std::string Description(const ColorEncoding& c_in) {
  // Copy required for Implicit*
  ColorEncoding c = c_in;

  std::string d = ToString(c.GetColorSpace());

  if (!c.ImplicitWhitePoint()) {
    d += '_';
    if (c.white_point == WhitePoint::kCustom) {
      const CIExy wp = c.GetWhitePoint();
      d += std::to_string(wp.x) + ';';
      d += std::to_string(wp.y);
    } else {
      d += ToString(c.white_point);
    }
  }

  if (c.HasPrimaries()) {
    d += '_';
    if (c.primaries == Primaries::kCustom) {
      const PrimariesCIExy pr = c.GetPrimaries();
      d += std::to_string(pr.r.x) + ';';
      d += std::to_string(pr.r.y) + ';';
      d += std::to_string(pr.g.x) + ';';
      d += std::to_string(pr.g.y) + ';';
      d += std::to_string(pr.b.x) + ';';
      d += std::to_string(pr.b.y);
    } else {
      d += ToString(c.primaries);
    }
  }

  d += '_';
  d += ToString(c.rendering_intent);

  if (!c.tf.SetImplicit()) {
    d += '_';
    if (c.tf.IsGamma()) {
      d += 'g';
      d += std::to_string(c.tf.GetGamma());
    } else {
      d += ToString(c.tf.GetTransferFunction());
    }
  }

  return d;
}

Status ParseDescription(const std::string& description,
                        ColorEncoding* JXL_RESTRICT c) {
  Tokenizer tokenizer(&description, '_');
  JXL_RETURN_IF_ERROR(ParseColorSpace(&tokenizer, c));
  JXL_RETURN_IF_ERROR(ParseWhitePoint(&tokenizer, c));
  JXL_RETURN_IF_ERROR(ParsePrimaries(&tokenizer, c));
  JXL_RETURN_IF_ERROR(ParseRenderingIntent(&tokenizer, c));
  JXL_RETURN_IF_ERROR(ParseTransferFunction(&tokenizer, c));
  return true;
}

Customxy::Customxy() { Bundle::Init(this); }
Status Customxy::VisitFields(Visitor* JXL_RESTRICT visitor) {
  uint32_t ux = PackSigned(x);
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Bits(19), BitsOffset(19, 524288),
                                         BitsOffset(20, 1048576),
                                         BitsOffset(21, 2097152), 0, &ux));
  x = UnpackSigned(ux);
  uint32_t uy = PackSigned(y);
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Bits(19), BitsOffset(19, 524288),
                                         BitsOffset(20, 1048576),
                                         BitsOffset(21, 2097152), 0, &uy));
  y = UnpackSigned(uy);
  return true;
}

CustomTransferFunction::CustomTransferFunction() { Bundle::Init(this); }
Status CustomTransferFunction::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->Conditional(!SetImplicit())) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &have_gamma_));

    if (visitor->Conditional(have_gamma_)) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(24, kGammaMul, &gamma_));
      if (gamma_ > kGammaMul) {
        return JXL_FAILURE("Invalid gamma %u", gamma_);
      }
    }

    if (visitor->Conditional(!have_gamma_)) {
      JXL_QUIET_RETURN_IF_ERROR(
          visitor->Enum(TransferFunction::kSRGB, &transfer_function_));
    }
  }

  return true;
}

ColorEncoding::ColorEncoding() { Bundle::Init(this); }
Status ColorEncoding::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &want_icc_));

  // Always send even if want_icc_ because this affects decoding.
  // We can skip the white point/primaries because they do not.
  JXL_QUIET_RETURN_IF_ERROR(visitor->Enum(ColorSpace::kRGB, &color_space_));

  if (visitor->Conditional(!WantICC())) {
    // Serialize enums. NOTE: we set the defaults to the most common values so
    // ImageMetadata.all_default is true in the common case.

    if (visitor->Conditional(!ImplicitWhitePoint())) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->Enum(WhitePoint::kD65, &white_point));
      if (visitor->Conditional(white_point == WhitePoint::kCustom)) {
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&white_));
      }
    }

    if (visitor->Conditional(HasPrimaries())) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->Enum(Primaries::kSRGB, &primaries));
      if (visitor->Conditional(primaries == Primaries::kCustom)) {
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&red_));
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&green_));
        JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&blue_));
      }
    }

    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&tf));

    JXL_QUIET_RETURN_IF_ERROR(
        visitor->Enum(RenderingIntent::kRelative, &rendering_intent));

    // We didn't have ICC, so all fields should be known.
    if (color_space_ == ColorSpace::kUnknown || tf.IsUnknown()) {
      return JXL_FAILURE("No ICC but cs %u and tf %u%s", color_space_,
                         tf.IsGamma() ? 0 : tf.GetTransferFunction(),
                         tf.IsGamma() ? "(gamma)" : "");
    }

    JXL_RETURN_IF_ERROR(CreateICC());
  }

  if (WantICC() && visitor->IsReading()) {
    // Haven't called SetICC() yet, do nothing.
  } else {
    if (ICC().empty()) return JXL_FAILURE("Empty ICC");
  }

  return true;
}

}  // namespace jxl
