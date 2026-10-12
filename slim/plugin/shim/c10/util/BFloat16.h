// Stand-in for c10/util/BFloat16.h (see Half.h).
#pragma once
#include <cuda_bf16.h>

namespace at {
using BFloat16 = __nv_bfloat16;
}
