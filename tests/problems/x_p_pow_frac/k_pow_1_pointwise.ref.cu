extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const float* pow_1_c1, const float* x, float* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x[a0 * 4];
        float in0__u0 = pow_1_c1[0];
        float v1__u0 = powf(fabsf(in1__u0), in0__u0);
        float in1__u1 = x[a0 * 4 + 1];
        float in0__u1 = pow_1_c1[0];
        float v1__u1 = powf(fabsf(in1__u1), in0__u1);
        float in1__u2 = x[a0 * 4 + 2];
        float in0__u2 = pow_1_c1[0];
        float v1__u2 = powf(fabsf(in1__u2), in0__u2);
        float in1__u3 = x[a0 * 4 + 3];
        float in0__u3 = pow_1_c1[0];
        float v1__u3 = powf(fabsf(in1__u3), in0__u3);
        float4 _vs_v1__u0_afc9565b = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&pow_1[a0 * 4]) = _vs_v1__u0_afc9565b;
}
