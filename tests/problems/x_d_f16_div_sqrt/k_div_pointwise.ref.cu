#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const __half* add_c1, const __half* x, __half* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        __half in1__u0 = x[a0 * 4];
        __half in0__u0 = add_c1[0];
        __half v3__u0 = in1__u0 / sqrtf((in0__u0 + fabsf(in1__u0)));
        __half in1__u1 = x[a0 * 4 + 1];
        __half in0__u1 = add_c1[0];
        __half v3__u1 = in1__u1 / sqrtf((in0__u1 + fabsf(in1__u1)));
        __half in1__u2 = x[a0 * 4 + 2];
        __half in0__u2 = add_c1[0];
        __half v3__u2 = in1__u2 / sqrtf((in0__u2 + fabsf(in1__u2)));
        __half in1__u3 = x[a0 * 4 + 3];
        __half in0__u3 = add_c1[0];
        __half v3__u3 = in1__u3 / sqrtf((in0__u3 + fabsf(in1__u3)));
        __half _vs_v3__u0_d87122d5[4] = { v3__u0, v3__u1, v3__u2, v3__u3 };
        *reinterpret_cast<uint2*>(&div[a0 * 4]) = *reinterpret_cast<const uint2*>(_vs_v3__u0_d87122d5);
}
