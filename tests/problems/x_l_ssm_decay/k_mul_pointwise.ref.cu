extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* mul_c1, const float* x0, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x0[a0 * 4];
        float in0__u0 = mul_c1[0];
        float v2__u0 = in0__u0 * (0.0f - fabsf(in1__u0));
        float in1__u1 = x0[a0 * 4 + 1];
        float in0__u1 = mul_c1[0];
        float v2__u1 = in0__u1 * (0.0f - fabsf(in1__u1));
        float in1__u2 = x0[a0 * 4 + 2];
        float in0__u2 = mul_c1[0];
        float v2__u2 = in0__u2 * (0.0f - fabsf(in1__u2));
        float in1__u3 = x0[a0 * 4 + 3];
        float in0__u3 = mul_c1[0];
        float v2__u3 = in0__u3 * (0.0f - fabsf(in1__u3));
        float4 _vs_v2__u0_30b35c22 = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 4]) = _vs_v2__u0_30b35c22;
}
