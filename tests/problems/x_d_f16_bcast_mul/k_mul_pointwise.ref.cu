#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const __half* x1, const __half* x0, __half* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 32;
        int a1 = (_gid) % 32;
        __half in0__u0 = x0[a0 * 128 + a1 * 4];
        __half in1__u0 = x1[a1 * 4];
        __half v0__u0 = in0__u0 * in1__u0;
        __half in0__u1 = x0[a0 * 128 + (a1 * 4 + 1)];
        __half in1__u1 = x1[a1 * 4 + 1];
        __half v0__u1 = in0__u1 * in1__u1;
        __half in0__u2 = x0[a0 * 128 + (a1 * 4 + 2)];
        __half in1__u2 = x1[a1 * 4 + 2];
        __half v0__u2 = in0__u2 * in1__u2;
        __half in0__u3 = x0[a0 * 128 + (a1 * 4 + 3)];
        __half in1__u3 = x1[a1 * 4 + 3];
        __half v0__u3 = in0__u3 * in1__u3;
        __half _vs_v0__u0_1303cb48[4] = { v0__u0, v0__u1, v0__u2, v0__u3 };
        *reinterpret_cast<uint2*>(&mul[a0 * 128 + a1 * 4]) = *reinterpret_cast<const uint2*>(_vs_v0__u0_1303cb48);
}
