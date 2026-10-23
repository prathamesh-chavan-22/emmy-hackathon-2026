extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const double* add_c1, const double* b, const double* a, double* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in2__u0 = b[a0 * 4];
        float in0__u0 = add_c1[0];
        float in1__u0 = a[a0 * 4];
        float v3__u0 = (in1__u0 * in2__u0) / (in0__u0 + fabsf(in2__u0));
        float in2__u1 = b[a0 * 4 + 1];
        float in0__u1 = add_c1[0];
        float in1__u1 = a[a0 * 4 + 1];
        float v3__u1 = (in1__u1 * in2__u1) / (in0__u1 + fabsf(in2__u1));
        float in2__u2 = b[a0 * 4 + 2];
        float in0__u2 = add_c1[0];
        float in1__u2 = a[a0 * 4 + 2];
        float v3__u2 = (in1__u2 * in2__u2) / (in0__u2 + fabsf(in2__u2));
        float in2__u3 = b[a0 * 4 + 3];
        float in0__u3 = add_c1[0];
        float in1__u3 = a[a0 * 4 + 3];
        float v3__u3 = (in1__u3 * in2__u3) / (in0__u3 + fabsf(in2__u3));
        div[a0 * 4] = v3__u0;
        div[a0 * 4 + 1] = v3__u1;
        div[a0 * 4 + 2] = v3__u2;
        div[a0 * 4 + 3] = v3__u3;
}
