extern "C" __global__
__launch_bounds__(256) void k_cumsum_reduce(const float* mul, float* cumsum) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 16) {
        int a0 = _gid;
        float acc0 = 0.0f;
        for (int a1 = 0; a1 < 256; a1++) {
            float in0 = mul[a0 * 256 + a1];
            acc0 += in0;
            float acc0__obs = acc0;
            cumsum[a0 * 256 + a1] = acc0__obs;
        }
    }
}
