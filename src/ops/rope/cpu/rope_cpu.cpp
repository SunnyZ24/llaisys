#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <bits/stdint-intn.h>
#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t Len, size_t MH, size_t D, float theta) {
    for (size_t i = 0; i < Len; i++) {
        for (size_t j = 0; j < MH; j++) {
            for (size_t k = 0; k < D; k++) {
                float temp = 0.0f;
                size_t j_pair = (k < D / 2) ? k : (k - D / 2);
                float rope_angle = pos_ids[i] / llaisys::utils::cast<float>(std::pow(theta, 2.0 * j_pair / D));
                // float rope_angle = pos_ids[i] / std::pow(theta, 2.0f * static_cast<float>(j_pair) / static_cast<float>(D));
                if (k < D / 2) {
                    temp = llaisys::utils::cast<float>(in[i * MH * D + j * D + k]) * std::cos(rope_angle) 
                    - llaisys::utils::cast<float>(in[i * MH * D + j * D + k + D / 2]) * std::sin(rope_angle);
                } else {
                    temp = llaisys::utils::cast<float>(in[i * MH * D + j * D + k]) * std::cos(rope_angle) 
                    + llaisys::utils::cast<float>(in[i * MH * D + j * D + k - D / 2]) * std::sin(rope_angle);
                }
                out[i * MH * D + j * D + k] = llaisys::utils::cast<T>(temp);
            }
        }   
    }
}
namespace llaisys::ops::cpu {

    void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
        llaisysDataType_t type, size_t Len, size_t MH, size_t D, float theta) {
      switch (type) {
      case LLAISYS_DTYPE_F32:
        return rope_<float>(reinterpret_cast<float *>(out),
                                reinterpret_cast<const float *>(in),
                                reinterpret_cast<const int64_t *>(pos_ids), Len, MH, D, theta);
      case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids), Len, MH, D, theta);
      case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const int64_t *>(pos_ids), Len, MH, D, theta);
      default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
      }
    }
    
    } // namespace llaisys::ops::cpu