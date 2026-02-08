#pragma once
#include "llaisys.h"

#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
// 1D argmax over `numel` elements in `vals`; writes max value into `max_val` and index into `max_idx`.
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel);
}