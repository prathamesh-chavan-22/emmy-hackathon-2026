extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const int* sub_c1, const int* x, int* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        int in1__u0 = x[a0 * 2];
        int in0__u0 = sub_c1[0];
        int v1__u0 = fabsf(in1__u0) - in0__u0;
        int in1__u1 = x[a0 * 2 + 1];
        int in0__u1 = sub_c1[0];
        int v1__u1 = fabsf(in1__u1) - in0__u1;
        sub[a0 * 2] = v1__u0;
        sub[a0 * 2 + 1] = v1__u1;
}
