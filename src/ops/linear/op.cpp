#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    size_t M = out->shape()[0];
    size_t N = out->shape()[1];
    size_t K = in->shape()[1];
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0] &&
                      in->shape()[1] == weight->shape()[1] && bias->shape()[0] == weight->shape()[0],
                  "Linear: shape mismatch");

    cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), M, N, K);
}

} // namespace llaisys::ops
