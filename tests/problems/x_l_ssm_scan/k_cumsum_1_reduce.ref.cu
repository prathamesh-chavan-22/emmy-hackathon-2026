extern "C" __global__
__launch_bounds__(256) void k_cumsum_1_reduce(const float* mul, float* cumsum_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float acc0 = 0.0f;
        for (int a0 = 0; a0 < 128; a0++) {
            float in0 = mul[a0];
            acc0 += in0;
            float acc0__obs = acc0;
            cumsum_1[a0] = acc0__obs;
        }
    }
}
