#include "linear_cpu.hpp"

#include "../../../utils.hpp"




template <typename T>
void vec_mat_mul_(std::byte *out, const std::byte *in, const std::byte *weight, size_t k, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float acc = 0.0f;
        for (size_t j = 0; j < k; j++) {
            acc += llaisys::utils::cast<const float>(reinterpret_cast<const T *>(in)[j]) * llaisys::utils::cast<const float>(reinterpret_cast<const T *>(weight)[i * k + j]);
        }
        reinterpret_cast<T *>(out)[i] = llaisys::utils::cast<T>(acc);
    }
     
} 


template <typename T>
void linear_(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; i++) {
        vec_mat_mul_<T>(out, in, weight, k, n);
        for (size_t j = 0; j < n; j++) {
            float temp = 0.0f;
            temp = llaisys::utils::cast<const float>(reinterpret_cast<const T *>(bias)[j]) + llaisys::utils::cast<const float>(reinterpret_cast<const T *>(out)[j]);
            reinterpret_cast<T *>(out)[j] = llaisys::utils::cast<T>(temp);
        }
        out += n * sizeof(T);
        in += k * sizeof(T);
    }
}

namespace llaisys::ops::cpu {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(out, in, weight, bias, M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(out, in, weight, bias, M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(out, in, weight, bias, M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu