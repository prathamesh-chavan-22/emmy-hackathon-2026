#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* mul_c1, const __half* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        __half in1__u0 = x[a0 * 2];
        float v0__u0 = __half2float(in1__u0);
        float in0__u0 = mul_c1[0];
        float v1__u0 = in0__u0 * v0__u0;
        __half in1__u1 = x[a0 * 2 + 1];
        float v0__u1 = __half2float(in1__u1);
        float in0__u1 = mul_c1[0];
        float v1__u1 = in0__u1 * v0__u1;
        float2 _vs_v1__u0_bcd8a33c = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&mul[a0 * 2]) = _vs_v1__u0_bcd8a33c;
}
