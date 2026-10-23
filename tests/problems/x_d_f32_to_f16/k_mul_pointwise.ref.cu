#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const __half* mul_c1, const float* x, __half* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 2];
        __half v0__u0 = __float2half(in1__u0);
        __half in0__u0 = mul_c1[0];
        __half v1__u0 = in0__u0 * v0__u0;
        float in1__u1 = x[a0 * 2 + 1];
        __half v0__u1 = __float2half(in1__u1);
        __half in0__u1 = mul_c1[0];
        __half v1__u1 = in0__u1 * v0__u1;
        __half2 _vs_v1__u0_bcd8a33c = __halves2half2(v1__u0, v1__u1);
        *reinterpret_cast<__half2*>(&mul[a0 * 2]) = _vs_v1__u0_bcd8a33c;
}
