extern "C" __global__
__launch_bounds__(128) void k_softmax_4e5397(const double* x, double* softmax) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = -1e+30f;
        for (int a1 = a1_co; a1 < 128; a1 += 128) {
            float in0 = x[a0 * 128 + a1];
            acc0 = fmaxf(acc0, in0);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = fmaxf(((float)(acc0)), acc0__o);
        }
        __shared__ float acc0_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = fmaxf(acc0, acc0__o);
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        float acc1 = 0.0f;
        for (int a1 = a1_co; a1 < 128; a1 += 128) {
            float in1 = x[a0 * 128 + a1];
            float v1 = expf((in1 - acc0));
            acc1 += v1;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            acc1 = acc1 + acc1__o;
        }
        __shared__ float acc1_smem[4];
        if (lane == 0) {
            acc1_smem[warp] = acc1;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                acc1 = acc1 + acc1__o;
                acc1_smem[warp] = acc1;
            }
            __syncthreads();
        }
        acc1 = acc1_smem[0];
        float v2 = 1.0f / acc1;
        for (int a3 = a1_co; a3 < 128; a3 += 128) {
            float in2 = x[a0 * 128 + a3];
            float v5 = v2 * expf((in2 - acc0));
            softmax[a0 * 128 + a3] = v5;
        }
}
