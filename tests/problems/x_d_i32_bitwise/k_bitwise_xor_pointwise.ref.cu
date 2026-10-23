extern "C" __global__
__launch_bounds__(256) void k_bitwise_xor_pointwise(const int* x1, const int* x0, int* bitwise_xor) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in0__u0 = x0[a0 * 2];
        int in1__u0 = x1[a0 * 2];
        int v2__u0 = (in0__u0 & in1__u0) ^ (in0__u0 | in1__u0);
        int in0__u1 = x0[a0 * 2 + 1];
        int in1__u1 = x1[a0 * 2 + 1];
        int v2__u1 = (in0__u1 & in1__u1) ^ (in0__u1 | in1__u1);
        bitwise_xor[a0 * 2] = v2__u0;
        bitwise_xor[a0 * 2 + 1] = v2__u1;
}
