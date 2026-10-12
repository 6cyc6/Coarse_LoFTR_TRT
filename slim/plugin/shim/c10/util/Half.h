// Stand-in for c10/util/Half.h so that VMamba's selective scan kernels build without torch: the kernels only use
// at::Half as a 2-byte storage type that converts to and from float and is reinterpreted as half2.
#pragma once
#include <cuda_fp16.h>

namespace at {
using Half = __half;
}
