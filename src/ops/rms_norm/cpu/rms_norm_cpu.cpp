#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t M, size_t N, float eps) {
    for (size_t i = 0; i < M; i++) {
        float avg = 0.0f;
        for (size_t k = 0; k < N; k++) {
            avg += llaisys::utils::cast<float>(in[i * N + k]) * llaisys::utils::cast<float>(in[i * N + k]);
        }
        avg /= N;
        float temp = 0.0f;
        for (size_t j = 0; j < N; j++) {
            temp = (llaisys::utils::cast<float>(in[i * N + j]) * llaisys::utils::cast<float>(weight[j])) / std::sqrt(eps + avg);
            out[i * N + j] = llaisys::utils::cast<T>(temp);
        }
    }
}

namespace llaisys::ops::cpu {

    void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, 
        llaisysDataType_t type, size_t M, size_t N, float eps) {
            switch (type) {
                case LLAISYS_DTYPE_F32:
                    return rms_norm_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), M, N, eps);
                case LLAISYS_DTYPE_BF16:
                    return rms_norm_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), M, N, eps);
                case LLAISYS_DTYPE_F16:
                    return rms_norm_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), M, N, eps);
                default:
                    EXCEPTION_UNSUPPORTED_DATATYPE(type);
                }
        } 

    } // namespace llaisys::ops::cpu
