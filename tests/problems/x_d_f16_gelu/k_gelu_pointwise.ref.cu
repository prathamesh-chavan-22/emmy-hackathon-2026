#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_gelu_pointwise(const __half* gelu_inv_sqrt2, const __half* gelu_one, const __half* gelu_half, const __half* x, __half* gelu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        __half in1__u0 = gelu_inv_sqrt2[0];
        __half in3__u0 = x[a0 * 4];
        __half in2__u0 = gelu_one[0];
        __half in0__u0 = gelu_half[0];
        __half v4__u0 = (in2__u0 + erff((in1__u0 * in3__u0))) * (in0__u0 * in3__u0);
        __half in1__u1 = gelu_inv_sqrt2[0];
        __half in3__u1 = x[a0 * 4 + 1];
        __half in2__u1 = gelu_one[0];
        __half in0__u1 = gelu_half[0];
        __half v4__u1 = (in2__u1 + erff((in1__u1 * in3__u1))) * (in0__u1 * in3__u1);
        __half in1__u2 = gelu_inv_sqrt2[0];
        __half in3__u2 = x[a0 * 4 + 2];
        __half in2__u2 = gelu_one[0];
        __half in0__u2 = gelu_half[0];
        __half v4__u2 = (in2__u2 + erff((in1__u2 * in3__u2))) * (in0__u2 * in3__u2);
        __half in1__u3 = gelu_inv_sqrt2[0];
        __half in3__u3 = x[a0 * 4 + 3];
        __half in2__u3 = gelu_one[0];
        __half in0__u3 = gelu_half[0];
        __half v4__u3 = (in2__u3 + erff((in1__u3 * in3__u3))) * (in0__u3 * in3__u3);
        __half _vs_v4__u0_dee486d2[4] = { v4__u0, v4__u1, v4__u2, v4__u3 };
        *reinterpret_cast<uint2*>(&gelu[a0 * 4]) = *reinterpret_cast<const uint2*>(_vs_v4__u0_dee486d2);
}
