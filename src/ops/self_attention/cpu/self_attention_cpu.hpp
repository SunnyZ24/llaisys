#pragma once
#include "llaisys.h"

#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
    void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, size_t qlen, size_t klen, size_t d, size_t nhead, size_t dv, size_t nkvh, float scale);
    // void linear(tensor_t out, tensor_t in, tensor_t weight);
}
    