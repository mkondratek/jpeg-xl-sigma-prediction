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

#ifndef JXL_DCT_UTIL_H_
#define JXL_DCT_UTIL_H_

#include <stddef.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/image.h"

namespace jxl {
using ac_qcoeff_t = float;
using ACImage = Plane<ac_qcoeff_t>;
using ACImage3 = Image3<ac_qcoeff_t>;
}  // namespace jxl

#endif  // JXL_DCT_UTIL_H_
