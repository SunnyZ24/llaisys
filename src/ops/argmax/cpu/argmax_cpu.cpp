#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(std::byte *out_max_idx, std::byte *out_max_val, const std::byte *vals, size_t numel) {
    if (numel == 0)
        return;

    const T *in = reinterpret_cast<const T *>(vals);
    T *out = reinterpret_cast<T *>(out_max_val);

    // init
    float best_f = llaisys::utils::cast<float>(in[0]);
    size_t best_idx = 0;

    for (size_t i = 1; i < numel; ++i) {
        float cur = llaisys::utils::cast<float>(in[i]);
        if (cur > best_f) {
            best_f = cur;
            best_idx = i;
        }
    }

    out[0] = llaisys::utils::cast<T>(best_f);
    // max_idx is int64
    // *reinterpret_cast<int64_t *>(out_max_idx) = static_cast<int64_t>(best_idx);
    reinterpret_cast<int64_t *>(out_max_idx)[0] = static_cast<int64_t>(best_idx);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_<llaisys::bf16_t>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return argmax_<llaisys::fp16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
