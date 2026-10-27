extern "C" __global__
__launch_bounds__(128) void k_matmul_reduce_280581(const float* silu_one, const float* x1, const float* x0, const float* x2, float* mul) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 32768;
        int a1 = (_gid / 128) % 256;
        int a2_co = (_gid) % 128;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int a2 = a2_co; a2 < 128; a2 += 128) {
            float in1 = x0[a0 * 128 + a2];
            float in2 = x1[a2 * 256 + a1];
            float v0 = in1 * in2;
            float in3 = x2[a2 * 256 + a1];
            float v1 = in1 * in3;
            acc0 += v0;
            acc1 += v1;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
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
                acc0 = acc0 + acc0__o;
                acc1 = acc1 + acc1__o;
                acc0_smem[warp] = acc0;
                acc1_smem[warp] = acc1;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        acc1 = acc1_smem[0];
        if (a2_co == 0) {
            float in0 = silu_one[0];
            float v7 = acc1 * (acc0 * (1.0f / (in0 + expf((0.0f - acc0)))));
            mul[a0 * 256 + a1] = v7;
        }
}
