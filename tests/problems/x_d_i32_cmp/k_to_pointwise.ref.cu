extern "C" __global__
__launch_bounds__(256) void k_to_pointwise(const int* x1, const int* x0, float* to) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in0__u0 = x0[a0 * 2];
        int in1__u0 = x1[a0 * 2];
        bool v0__u0 = in0__u0 > in1__u0;
        int in0__u1 = x0[a0 * 2 + 1];
        int in1__u1 = x1[a0 * 2 + 1];
        bool v0__u1 = in0__u1 > in1__u1;
        float2 _vs_v0__u0_b6d35f79 = make_float2(v0__u0, v0__u1);
        *reinterpret_cast<float2*>(&to[a0 * 2]) = _vs_v0__u0_b6d35f79;
}
