#pragma once
#include "llaisys.h"

#include <cstddef>
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
    void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t seqlen, size_t intermediate_size);
}