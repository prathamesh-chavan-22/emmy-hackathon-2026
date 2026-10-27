#include <cuda_fp8.h>
extern "C" __global__
__launch_bounds__(256) void k_clamp_pointwise(const float* mul_c1, const float* clamp_bound0, const float* clamp_bound1, const float* x, float* clamp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = mul_c1[0];
        float in3__u0 = x[a0 * 4];
        __nv_fp8_e5m2 v1__u0 = __nv_fp8_e5m2((in2__u0 * in3__u0));
        float in0__u0 = clamp_bound0[0];
        float v2__u0 = fmaxf(in0__u0, float(v1__u0));
        float in1__u0 = clamp_bound1[0];
        float v3__u0 = fminf(in1__u0, v2__u0);
        float in2__u1 = mul_c1[0];
        float in3__u1 = x[a0 * 4 + 1];
        __nv_fp8_e5m2 v1__u1 = __nv_fp8_e5m2((in2__u1 * in3__u1));
        float in0__u1 = clamp_bound0[0];
        float v2__u1 = fmaxf(in0__u1, float(v1__u1));
        float in1__u1 = clamp_bound1[0];
        float v3__u1 = fminf(in1__u1, v2__u1);
        float in2__u2 = mul_c1[0];
        float in3__u2 = x[a0 * 4 + 2];
        __nv_fp8_e5m2 v1__u2 = __nv_fp8_e5m2((in2__u2 * in3__u2));
        float in0__u2 = clamp_bound0[0];
        float v2__u2 = fmaxf(in0__u2, float(v1__u2));
        float in1__u2 = clamp_bound1[0];
        float v3__u2 = fminf(in1__u2, v2__u2);
        float in2__u3 = mul_c1[0];
        float in3__u3 = x[a0 * 4 + 3];
        __nv_fp8_e5m2 v1__u3 = __nv_fp8_e5m2((in2__u3 * in3__u3));
        float in0__u3 = clamp_bound0[0];
        float v2__u3 = fmaxf(in0__u3, float(v1__u3));
        float in1__u3 = clamp_bound1[0];
        float v3__u3 = fminf(in1__u3, v2__u3);
        float4 _vs_v3__u0_21dbbd11 = make_float4(v3__u0, v3__u1, v3__u2, v3__u3);
        *reinterpret_cast<float4*>(&clamp[a0 * 4]) = _vs_v3__u0_21dbbd11;
}
