#pragma once

#include "cpu/self_attention_cpu.hpp"

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
}
