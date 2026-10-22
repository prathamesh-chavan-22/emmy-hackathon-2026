extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* softplus_one, const float* softplus_zero, const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = x[a0 * 4];
        float in1__u0 = softplus_zero[0];
        float in0__u0 = softplus_one[0];
        float v8__u0 = in2__u0 * tanhf((fmaxf(in1__u0, in2__u0) + log((in0__u0 + expf((0.0f - fabsf(in2__u0)))))));
        float in2__u1 = x[a0 * 4 + 1];
        float in1__u1 = softplus_zero[0];
        float in0__u1 = softplus_one[0];
        float v8__u1 = in2__u1 * tanhf((fmaxf(in1__u1, in2__u1) + log((in0__u1 + expf((0.0f - fabsf(in2__u1)))))));
        float in2__u2 = x[a0 * 4 + 2];
        float in1__u2 = softplus_zero[0];
        float in0__u2 = softplus_one[0];
        float v8__u2 = in2__u2 * tanhf((fmaxf(in1__u2, in2__u2) + log((in0__u2 + expf((0.0f - fabsf(in2__u2)))))));
        float in2__u3 = x[a0 * 4 + 3];
        float in1__u3 = softplus_zero[0];
        float in0__u3 = softplus_one[0];
        float v8__u3 = in2__u3 * tanhf((fmaxf(in1__u3, in2__u3) + log((in0__u3 + expf((0.0f - fabsf(in2__u3)))))));
        float4 _vs_v8__u0_3054d580 = make_float4(v8__u0, v8__u1, v8__u2, v8__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 4]) = _vs_v8__u0_3054d580;
}
