extern "C" __global__
__launch_bounds__(128) void k_sub_2_reduce(const float* x, float* sub_2) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = -1e+30f;
        float acc1 = -1e+30f;
        for (int a1 = a1_co; a1 < 200; a1 += 128) {
            float in0 = x[a0 * 200 + a1];
            acc0 = fmaxf(acc0, in0);
            acc1 = fmaxf(acc1, in0);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            acc0 = fmaxf(acc0, acc0__o);
            acc1 = fmaxf(acc1, acc1__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            acc0 = fmaxf(acc0, acc0__o);
            acc1 = fmaxf(acc1, acc1__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            acc0 = fmaxf(acc0, acc0__o);
            acc1 = fmaxf(acc1, acc1__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            acc0 = fmaxf(acc0, acc0__o);
            acc1 = fmaxf(acc1, acc1__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            acc0 = fmaxf(acc0, acc0__o);
            acc1 = fmaxf(acc1, acc1__o);
        }
        __shared__ float acc0_smem[4];
        __shared__ float acc1_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
            acc1_smem[warp] = acc1;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                acc0 = fmaxf(acc0, acc0__o);
                acc1 = fmaxf(acc1, acc1__o);
                acc0_smem[warp] = acc0;
                acc1_smem[warp] = acc1;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        acc1 = acc1_smem[0];
        float acc2 = 0.0f;
        for (int a1 = a1_co; a1 < 200; a1 += 128) {
            float in1 = x[a0 * 200 + a1];
            float v1 = expf((in1 - acc1));
            acc2 += v1;
        }
        {
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 16);
            acc2 = acc2 + acc2__o;
        }
        {
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 8);
            acc2 = acc2 + acc2__o;
        }
        {
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 4);
            acc2 = acc2 + acc2__o;
        }
        {
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 2);
            acc2 = acc2 + acc2__o;
        }
        {
            float acc2__o = __shfl_xor_sync(__activemask(), acc2, 1);
            acc2 = acc2 + acc2__o;
        }
        __shared__ float acc2_smem[4];
        if (lane == 0) {
            acc2_smem[warp] = acc2;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc2 = acc2_smem[warp];
                float acc2__o = acc2_smem[warp + s];
                acc2 = acc2 + acc2__o;
                acc2_smem[warp] = acc2;
            }
            __syncthreads();
        }
        acc2 = acc2_smem[0];
        float v2 = log(acc2);
        for (int a3 = a1_co; a3 < 200; a3 += 128) {
            float in2 = x[a0 * 200 + a3];
            float v4 = (in2 - acc0) - v2;
            sub_2[a0 * 200 + a3] = v4;
        }
}
