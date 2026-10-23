extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const long long* x1, const long long* x0, long long* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        long long in0__u0 = x0[a0 * 2];
        long long in1__u0 = x1[a0 * 2];
        long long v1__u0 = in1__u0 + (in0__u0 * in1__u0);
        long long in0__u1 = x0[a0 * 2 + 1];
        long long in1__u1 = x1[a0 * 2 + 1];
        long long v1__u1 = in1__u1 + (in0__u1 * in1__u1);
        add[a0 * 2] = v1__u0;
        add[a0 * 2 + 1] = v1__u1;
}
