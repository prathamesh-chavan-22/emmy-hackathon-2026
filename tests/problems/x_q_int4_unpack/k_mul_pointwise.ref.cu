extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const int* rshift_c1, const int* sub_c1, const int* and_1_c1, const float* mul_c1, const int* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in2__u0 = rshift_c1[0];
        int in4__u0 = x[a0 * 4];
        int in0__u0 = and_1_c1[0];
        int in3__u0 = sub_c1[0];
        float in1__u0 = mul_c1[0];
        float v3__u0 = in1__u0 * (((in4__u0 >> in2__u0) & in0__u0) - in3__u0);
        int in2__u1 = rshift_c1[0];
        int in4__u1 = x[a0 * 4 + 1];
        int in0__u1 = and_1_c1[0];
        int in3__u1 = sub_c1[0];
        float in1__u1 = mul_c1[0];
        float v3__u1 = in1__u1 * (((in4__u1 >> in2__u1) & in0__u1) - in3__u1);
        int in2__u2 = rshift_c1[0];
        int in4__u2 = x[a0 * 4 + 2];
        int in0__u2 = and_1_c1[0];
        int in3__u2 = sub_c1[0];
        float in1__u2 = mul_c1[0];
        float v3__u2 = in1__u2 * (((in4__u2 >> in2__u2) & in0__u2) - in3__u2);
        int in2__u3 = rshift_c1[0];
        int in4__u3 = x[a0 * 4 + 3];
        int in0__u3 = and_1_c1[0];
        int in3__u3 = sub_c1[0];
        float in1__u3 = mul_c1[0];
        float v3__u3 = in1__u3 * (((in4__u3 >> in2__u3) & in0__u3) - in3__u3);
        float4 _vs_v3__u0_d587d722 = make_float4(v3__u0, v3__u1, v3__u2, v3__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 4]) = _vs_v3__u0_d587d722;
}
