#pragma once
#include "llaisys.h"

#include <cstddef>
#include "../../../tensor/tensor.hpp"


namespace llaisys::ops::cpu {
    void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
        llaisysDataType_t type, size_t Len, size_t MH, size_t D, float theta); 
}