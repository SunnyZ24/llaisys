#include "op.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    size_t seqlen = out->shape()[0];
    size_t intermediate_size = out->shape()[1];

    CHECK_ARGUMENT(seqlen == gate->shape()[0] && seqlen == up->shape()[0] && seqlen == out->shape()[0]
    && intermediate_size == gate->shape()[1] && intermediate_size == up->shape()[1] && intermediate_size == out->shape()[1], 
    "Invalid shape");

    cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), seqlen, intermediate_size);
}
} // namespace llaisys::ops
