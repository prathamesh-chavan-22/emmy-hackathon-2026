extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* div_c1, const int* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = div_c1[0];
        int in1__u0 = x[a0 * 2];
        float v1__u0 = in1__u0 * (1.0f / in0__u0);
        float in0__u1 = div_c1[0];
        int in1__u1 = x[a0 * 2 + 1];
        float v1__u1 = in1__u1 * (1.0f / in0__u1);
        float2 _vs_v1__u0_c4a7abae = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&div[a0 * 2]) = _vs_v1__u0_c4a7abae;
}
