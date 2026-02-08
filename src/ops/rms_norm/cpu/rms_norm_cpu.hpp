#pragma once
#include "llaisys.h"

#include <cstddef>
#include "../../../tensor/tensor.hpp"


namespace llaisys::ops::cpu {
    void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
        llaisysDataType_t type, size_t M, size_t N, float eps); 
}