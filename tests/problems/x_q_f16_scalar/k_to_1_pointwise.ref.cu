#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_to_1_pointwise(const __half* mul_c1, const float* x, float* to_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 4];
        __half v0__u0 = __float2half(in1__u0);
        __half in0__u0 = mul_c1[0];
        float v2__u0 = __half2float((in0__u0 * v0__u0));
        float in1__u1 = x[a0 * 4 + 1];
        __half v0__u1 = __float2half(in1__u1);
        __half in0__u1 = mul_c1[0];
        float v2__u1 = __half2float((in0__u1 * v0__u1));
        float in1__u2 = x[a0 * 4 + 2];
        __half v0__u2 = __float2half(in1__u2);
        __half in0__u2 = mul_c1[0];
        float v2__u2 = __half2float((in0__u2 * v0__u2));
        float in1__u3 = x[a0 * 4 + 3];
        __half v0__u3 = __float2half(in1__u3);
        __half in0__u3 = mul_c1[0];
        float v2__u3 = __half2float((in0__u3 * v0__u3));
        float4 _vs_v2__u0_2ab67ca3 = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&to_1[a0 * 4]) = _vs_v2__u0_2ab67ca3;
}
