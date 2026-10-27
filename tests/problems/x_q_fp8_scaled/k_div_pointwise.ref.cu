#include <cuda_fp8.h>
extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* mul_c1, const float* div_c1, const float* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = div_c1[0];
        float in1__u0 = mul_c1[0];
        float in2__u0 = x[a0 * 4];
        __nv_fp8_e4m3 v2__u0 = __nv_fp8_e4m3((in1__u0 * in2__u0));
        float v3__u0 = (1.0f / in0__u0) * float(v2__u0);
        float in0__u1 = div_c1[0];
        float in1__u1 = mul_c1[0];
        float in2__u1 = x[a0 * 4 + 1];
        __nv_fp8_e4m3 v2__u1 = __nv_fp8_e4m3((in1__u1 * in2__u1));
        float v3__u1 = (1.0f / in0__u1) * float(v2__u1);
        float in0__u2 = div_c1[0];
        float in1__u2 = mul_c1[0];
        float in2__u2 = x[a0 * 4 + 2];
        __nv_fp8_e4m3 v2__u2 = __nv_fp8_e4m3((in1__u2 * in2__u2));
        float v3__u2 = (1.0f / in0__u2) * float(v2__u2);
        float in0__u3 = div_c1[0];
        float in1__u3 = mul_c1[0];
        float in2__u3 = x[a0 * 4 + 3];
        __nv_fp8_e4m3 v2__u3 = __nv_fp8_e4m3((in1__u3 * in2__u3));
        float v3__u3 = (1.0f / in0__u3) * float(v2__u3);
        float4 _vs_v3__u0_d87122d5 = make_float4(v3__u0, v3__u1, v3__u2, v3__u3);
        *reinterpret_cast<float4*>(&div[a0 * 4]) = _vs_v3__u0_d87122d5;
}
