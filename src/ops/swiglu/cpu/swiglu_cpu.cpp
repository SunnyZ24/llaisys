#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t seqlen, size_t intermediate_size) {
    for (size_t i = 0; i < seqlen; i++) {
        // out[i] = up[i] * gate[i] / (1 + std::exp(-gate[i]));
        float temp = 0.0f;
        for (size_t j = 0; j < intermediate_size; j++) {
            temp = llaisys::utils::cast<float>(up[i * intermediate_size + j]) * llaisys::utils::cast<float>(gate[i * intermediate_size + j]) / (1 + std::exp(-llaisys::utils::cast<float>(gate[i * intermediate_size + j])));
            out[i * intermediate_size + j] = llaisys::utils::cast<T>(temp);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t seqlen, size_t intermediate_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), seqlen, intermediate_size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate), reinterpret_cast<const llaisys::bf16_t *>(up), seqlen, intermediate_size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate), reinterpret_cast<const llaisys::fp16_t *>(up), seqlen, intermediate_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu