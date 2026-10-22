extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* mul_c1, const float* mul_2_c1, const float* x, const float* y, const float* z, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in1__u0 = mul_c1[0];
        float in2__u0 = x[a0 * 4];
        float in0__u0 = mul_2_c1[0];
        float in3__u0 = y[a0 * 4];
        float in4__u0 = z[a0 * 4];
        float v4__u0 = (in4__u0 * (in3__u0 + (in1__u0 * in2__u0))) - (in0__u0 * in2__u0);
        float in1__u1 = mul_c1[0];
        float in2__u1 = x[a0 * 4 + 1];
        float in0__u1 = mul_2_c1[0];
        float in3__u1 = y[a0 * 4 + 1];
        float in4__u1 = z[a0 * 4 + 1];
        float v4__u1 = (in4__u1 * (in3__u1 + (in1__u1 * in2__u1))) - (in0__u1 * in2__u1);
        float in1__u2 = mul_c1[0];
        float in2__u2 = x[a0 * 4 + 2];
        float in0__u2 = mul_2_c1[0];
        float in3__u2 = y[a0 * 4 + 2];
        float in4__u2 = z[a0 * 4 + 2];
        float v4__u2 = (in4__u2 * (in3__u2 + (in1__u2 * in2__u2))) - (in0__u2 * in2__u2);
        float in1__u3 = mul_c1[0];
        float in2__u3 = x[a0 * 4 + 3];
        float in0__u3 = mul_2_c1[0];
        float in3__u3 = y[a0 * 4 + 3];
        float in4__u3 = z[a0 * 4 + 3];
        float v4__u3 = (in4__u3 * (in3__u3 + (in1__u3 * in2__u3))) - (in0__u3 * in2__u3);
        float4 _vs_v4__u0_0a07631b = make_float4(v4__u0, v4__u1, v4__u2, v4__u3);
        *reinterpret_cast<float4*>(&sub[a0 * 4]) = _vs_v4__u0_0a07631b;
}
