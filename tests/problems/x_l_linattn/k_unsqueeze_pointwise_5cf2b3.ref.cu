extern "C" __global__
__launch_bounds__(256) void k_unsqueeze_pointwise_5cf2b3(const float* x1, const float* x2, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 64;
        int a1 = (_gid / 4) % 16;
        int a2 = (_gid) % 4;
        float in0__u0 = x1[a0 * 16 + a1];
        float in1__u0 = x2[a0 * 16 + a2 * 4];
        float v0__u0 = in0__u0 * in1__u0;
        float in0__u1 = x1[a0 * 16 + a1];
        float in1__u1 = x2[a0 * 16 + (a2 * 4 + 1)];
        float v0__u1 = in0__u1 * in1__u1;
        float in0__u2 = x1[a0 * 16 + a1];
        float in1__u2 = x2[a0 * 16 + (a2 * 4 + 2)];
        float v0__u2 = in0__u2 * in1__u2;
        float in0__u3 = x1[a0 * 16 + a1];
        float in1__u3 = x2[a0 * 16 + (a2 * 4 + 3)];
        float v0__u3 = in0__u3 * in1__u3;
        float4 _vs_v0__u0_88ae57dc = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 256 + a1 * 16 + a2 * 4]) = _vs_v0__u0_88ae57dc;
}
