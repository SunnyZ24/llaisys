#include "embedding_cpu.hpp"

#include "../../../utils.hpp"


template <typename T>
void embedding_(llaisys::tensor_t &out, const llaisys::tensor_t &index, const llaisys::tensor_t &weight) {
    size_t numel = index->numel();
    if (numel == 0) return;
    const T *in_weight = reinterpret_cast<const T *>(weight->data());
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(index->data());
    T *out_ptr = reinterpret_cast<T *>(out->data());
    size_t E = weight->shape()[1]; // 长度
    ptrdiff_t out_stride = out->strides()[0];//测试 size_t 也可以，但是最好还是使用 ptrdiff
    for (size_t i = 0; i < index->numel(); i++) {
        for (size_t j = 0; j < E; j++) {
            out_ptr[i * out_stride + j] = in_weight[idx_ptr[i] * out_stride + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(tensor_t out, const tensor_t index, const tensor_t weight) {
    llaisysDataType_t type = weight->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(out, index, weight);
    case LLAISYS_DTYPE_BF16:
        return embedding_<llaisys::bf16_t>(out, index, weight);
    case LLAISYS_DTYPE_F16:
        return embedding_<llaisys::fp16_t>(out, index, weight);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
