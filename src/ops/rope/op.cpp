#include "op.hpp"
#include <cstddef>
#include "../../core/llaisys_core.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1]
                    && out->shape()[2] == in->shape()[2] && out->dtype() == in->dtype()
                    && pos_ids->shape()[0] == in->shape()[0] && pos_ids->shape()[0] % 2 == 0,
                     "Rope: shape mismatch");
    size_t Len = in->shape()[0];
    size_t Mhead = in->shape()[1];
    size_t D = in->shape()[2];
    cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), Len, Mhead, D, theta);
    
}
} // namespace llaisys::ops
