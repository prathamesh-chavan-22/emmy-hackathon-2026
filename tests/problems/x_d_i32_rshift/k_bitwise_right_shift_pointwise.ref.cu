extern "C" __global__
__launch_bounds__(256) void k_bitwise_right_shift_pointwise(const int* bitwise_right_shift_c1, const int* x, int* bitwise_right_shift) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in0__u0 = bitwise_right_shift_c1[0];
        int in1__u0 = x[a0 * 2];
        int v0__u0 = in1__u0 >> in0__u0;
        int in0__u1 = bitwise_right_shift_c1[0];
        int in1__u1 = x[a0 * 2 + 1];
        int v0__u1 = in1__u1 >> in0__u1;
        bitwise_right_shift[a0 * 2] = v0__u0;
        bitwise_right_shift[a0 * 2 + 1] = v0__u1;
}
