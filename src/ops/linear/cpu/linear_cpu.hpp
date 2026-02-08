#pragma once
#include "llaisys.h"

#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
// 1D argmax over `numel` elements in `vals`; writes max value into `max_val` and index into `max_idx`.
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K); 
}