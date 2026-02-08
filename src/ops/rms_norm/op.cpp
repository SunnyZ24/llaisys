#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"


namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t M = out->shape()[0];
    size_t N = out->shape()[1];
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1] &&
                      weight->shape()[0] == in->shape()[1] && out->dtype() == in->dtype() 
    && out->dtype() == weight->dtype(), "RMS Norm: shape mismatch");


    cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), M, N, eps);
}
} // namespace llaisys::ops
