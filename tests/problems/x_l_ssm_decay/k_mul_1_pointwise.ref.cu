extern "C" __global__
__launch_bounds__(256) void k_mul_1_pointwise(const float* cumsum, const float* x1, float* mul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = cumsum[a0 * 4];
        float in1__u0 = x1[a0 * 4];
        float v1__u0 = in1__u0 * expf(in0__u0);
        float in0__u1 = cumsum[a0 * 4 + 1];
        float in1__u1 = x1[a0 * 4 + 1];
        float v1__u1 = in1__u1 * expf(in0__u1);
        float in0__u2 = cumsum[a0 * 4 + 2];
        float in1__u2 = x1[a0 * 4 + 2];
        float v1__u2 = in1__u2 * expf(in0__u2);
        float in0__u3 = cumsum[a0 * 4 + 3];
        float in1__u3 = x1[a0 * 4 + 3];
        float v1__u3 = in1__u3 * expf(in0__u3);
        float4 _vs_v1__u0_995bda48 = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&mul_1[a0 * 4]) = _vs_v1__u0_995bda48;
}
