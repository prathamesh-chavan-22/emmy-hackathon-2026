extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const float* gt_c1, const int* x, int* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = gt_c1[0];
        int in1__u0 = x[a0 * 2];
        bool v0__u0 = in1__u0 > in0__u0;
        int v2__u0 = ((v0__u0) ? (in1__u0) : ((0.0f - in1__u0)));
        float in0__u1 = gt_c1[0];
        int in1__u1 = x[a0 * 2 + 1];
        bool v0__u1 = in1__u1 > in0__u1;
        int v2__u1 = ((v0__u1) ? (in1__u1) : ((0.0f - in1__u1)));
        where[a0 * 2] = v2__u0;
        where[a0 * 2 + 1] = v2__u1;
}
