extern "C" __global__
__launch_bounds__(256) void k_to_pointwise(const float* mul_c1, const float* x, int* to) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = mul_c1[0];
        float in1__u0 = x[a0 * 2];
        int v1__u0 = (in0__u0 * in1__u0);
        float in0__u1 = mul_c1[0];
        float in1__u1 = x[a0 * 2 + 1];
        int v1__u1 = (in0__u1 * in1__u1);
        to[a0 * 2] = v1__u0;
        to[a0 * 2 + 1] = v1__u1;
}
