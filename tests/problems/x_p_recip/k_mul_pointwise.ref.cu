extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* add_c1, const float* mul_c1, const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = x[a0 * 4];
        float in0__u0 = add_c1[0];
        float in1__u0 = mul_c1[0];
        float v3__u0 = in1__u0 * (1.0f / (in0__u0 + fabsf(in2__u0)));
        float in2__u1 = x[a0 * 4 + 1];
        float in0__u1 = add_c1[0];
        float in1__u1 = mul_c1[0];
        float v3__u1 = in1__u1 * (1.0f / (in0__u1 + fabsf(in2__u1)));
        float in2__u2 = x[a0 * 4 + 2];
        float in0__u2 = add_c1[0];
        float in1__u2 = mul_c1[0];
        float v3__u2 = in1__u2 * (1.0f / (in0__u2 + fabsf(in2__u2)));
        float in2__u3 = x[a0 * 4 + 3];
        float in0__u3 = add_c1[0];
        float in1__u3 = mul_c1[0];
        float v3__u3 = in1__u3 * (1.0f / (in0__u3 + fabsf(in2__u3)));
        float4 _vs_v3__u0_d587d722 = make_float4(v3__u0, v3__u1, v3__u2, v3__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 4]) = _vs_v3__u0_d587d722;
}
