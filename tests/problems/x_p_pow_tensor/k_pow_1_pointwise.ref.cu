extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const float* add_c1, const float* x0, const float* x1, float* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = x0[a0 * 4];
        float in0__u0 = add_c1[0];
        float in2__u0 = x1[a0 * 4];
        float v2__u0 = powf((in0__u0 + fabsf(in1__u0)), in2__u0);
        float in1__u1 = x0[a0 * 4 + 1];
        float in0__u1 = add_c1[0];
        float in2__u1 = x1[a0 * 4 + 1];
        float v2__u1 = powf((in0__u1 + fabsf(in1__u1)), in2__u1);
        float in1__u2 = x0[a0 * 4 + 2];
        float in0__u2 = add_c1[0];
        float in2__u2 = x1[a0 * 4 + 2];
        float v2__u2 = powf((in0__u2 + fabsf(in1__u2)), in2__u2);
        float in1__u3 = x0[a0 * 4 + 3];
        float in0__u3 = add_c1[0];
        float in2__u3 = x1[a0 * 4 + 3];
        float v2__u3 = powf((in0__u3 + fabsf(in1__u3)), in2__u3);
        float4 _vs_v2__u0_05e0bc4a = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&pow_1[a0 * 4]) = _vs_v2__u0_05e0bc4a;
}
