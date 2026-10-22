extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* y, const float* x, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = x[a0 * 4];
        float in1__u0 = y[a0 * 4];
        float v2__u0 = fmaxf(in0__u0, in1__u0) - fminf(in0__u0, in1__u0);
        float in0__u1 = x[a0 * 4 + 1];
        float in1__u1 = y[a0 * 4 + 1];
        float v2__u1 = fmaxf(in0__u1, in1__u1) - fminf(in0__u1, in1__u1);
        float in0__u2 = x[a0 * 4 + 2];
        float in1__u2 = y[a0 * 4 + 2];
        float v2__u2 = fmaxf(in0__u2, in1__u2) - fminf(in0__u2, in1__u2);
        float in0__u3 = x[a0 * 4 + 3];
        float in1__u3 = y[a0 * 4 + 3];
        float v2__u3 = fmaxf(in0__u3, in1__u3) - fminf(in0__u3, in1__u3);
        float4 _vs_v2__u0_05a843af = make_float4(v2__u0, v2__u1, v2__u2, v2__u3);
        *reinterpret_cast<float4*>(&sub[a0 * 4]) = _vs_v2__u0_05a843af;
}
