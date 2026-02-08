#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention__(T *attn_val, const T *q, const T *k, const T *v, size_t qlen, size_t klen, size_t d, size_t nhead, size_t dv, size_t nkvh, float scale) {
    std::vector<float> logits(qlen * nhead * klen, -std::numeric_limits<float>::infinity());

    // A = Q @ K^T * scale, with causal mask fused: j > i + (klen - qlen) -> -inf
    size_t causal_off = (klen >= qlen) ? (klen - qlen) : 0;
    for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < nhead; h++) {
            size_t kvh = h * nkvh / nhead;
            for (size_t j = 0; j < klen; j++) {
                if (j > i + causal_off) {
                    // logits[(i * nhead + h) * klen + j] = -std::numeric_limits<float>::infinity();
                    continue;
                } else {
                    float acc = 0.0f;
                    for (size_t dd = 0; dd < d; dd++) {
                        float qv = llaisys::utils::cast<float>(q[(i * nhead + h) * d + dd]);
                        float kv = llaisys::utils::cast<float>(k[(j * nkvh + kvh) * d + dd]);
                        acc += qv * kv;
                    }
                    logits[(i * nhead + h) * klen + j] = acc * scale;
                }
            }
        }
    }

    // softmax over j, then Y = P @ V (fused per (i,h))
    for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < nhead; h++) {
            float *row = &logits[(i * nhead + h) * klen];
            float mx = row[0];
            for (size_t j = 1; j < klen; j++) {
                if (row[j] > mx)
                    mx = row[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j < klen; j++) {
                row[j] = std::exp(row[j] - mx);
                sum += row[j];
            }
            for (size_t j = 0; j < klen; j++) {
                row[j] /= sum;
            }

            // Y[i,h,:] = P[i,h,:] @ V[:,kvh,:]
            // 这里乘的 V 没有转置,因此 dimension 在外面
            size_t kvh = h * nkvh / nhead;
            for (size_t kk = 0; kk < dv; kk++) {
                float acc = 0.0f;
                for (size_t j = 0; j < klen; j++) {
                    acc += row[j] * llaisys::utils::cast<float>(v[(j * nkvh + kvh) * dv + kk]);
                }
                attn_val[(i * nhead + h) * dv + kk] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, size_t qlen, size_t klen, size_t d, size_t nhead, size_t dv, size_t nkvh, float scale) {
    // logits is A and already be masked
    std::vector<float> logits(qlen * nhead * klen, -std::numeric_limits<float>::infinity());
    size_t causal_off = (klen >= qlen) ? (klen - qlen) : 0;

    // A = Q @ K^T * scale, with causal mask fused: j > i + (klen - qlen) -> -inf
    for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < nhead; h++) {
            size_t kvh = h * nkvh / nhead;
            for (size_t j = 0; j < klen; j++) {
                float acc = 0.0f;
                if (j <= i + causal_off) {
                    for (size_t dd = 0; dd < d; dd++) {
                        acc += llaisys::utils::cast<float>(q[(i * nhead + h) * d + dd]) * llaisys::utils::cast<float>(k[(j * nkvh + kvh) * d + dd]);
                    }
                    logits[(i * nhead + h) * klen + j] = acc * scale;
                } else break;
            }
        }
    }

    for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < nhead; h ++) {
            // softmax over j
            float *row = &logits[(i * nhead + h) * klen];
            float mx = row[0];
            for (size_t j = 1; j < klen; j++) {
                if (row[j] > mx)
                    mx = row[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j < klen; j++) {
                row[j] = std::exp(row[j] - mx);
                sum += row[j];
            }
            for (size_t j = 0; j < klen; j ++)  {
                row[j] /= sum;
            }

            // softmax(A) @ V
            for (size_t kk = 0; kk < dv; kk ++) {
                size_t kvh = h * nkvh / nhead;
                float acc = 0.0f;
                for (size_t j = 0; j < klen; j ++) {
                    acc += row[j] * llaisys::utils::cast<float>(v[(j * nkvh + kvh) * dv + kk]);
                }
                attn_val[(i * nhead + h) * dv + kk] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, size_t qlen, size_t klen, size_t d, size_t nhead, size_t dv, size_t nkvh, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), qlen, klen, d, nhead, dv, nkvh, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), qlen, klen, d, nhead, dv, nkvh, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), qlen, klen, d, nhead, dv, nkvh, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu