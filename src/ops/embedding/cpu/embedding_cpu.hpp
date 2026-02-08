#pragma once
#include "llaisys.h"

#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
// 1D argmax over `numel` elements in `vals`; writes max value into `max_val` and index into `max_idx`.
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}