#include "op.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    size_t qlen = q->shape()[0];
    size_t klen = k->shape()[0];
    size_t vlen = v->shape()[0];
    size_t nhead = q->shape()[1];
    size_t nkvh = k->shape()[1];
    size_t d = q->shape()[2];
    size_t dv = v->shape()[2];

    CHECK_ARGUMENT(d == k->shape()[2] && klen == vlen 
    && nkvh == v->shape()[1] && nhead == q->shape()[1]
    && dv == attn_val->shape()[2], "qlen must be equal to klen");

    cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen, klen, d, nhead, dv, nkvh, scale);

}
} // namespace llaisys::ops
