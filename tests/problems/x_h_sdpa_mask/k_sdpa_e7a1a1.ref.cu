#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(128) void k_sdpa_e7a1a1(const __half* scaled_dot_product_attention_scale, const __half* x1, const __half* x0, const __half* x3, const __half* x2, __half* scaled_dot_product_attention) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 262144;
        int a1 = (_gid / 4096) % 64;
        int a6 = (_gid / 128) % 32;
        int a2_co = (_gid) % 128;
        __half in0 = scaled_dot_product_attention_scale[0];
        float acc3 = 0.0f;
        float acc5__sum = 0.0f;
        float acc1 = -1e+30f;
        for (int a2 = a2_co; a2 < 64; a2 += 128) {
            float acc0 = 0.0f;
            for (int a3 = 0; a3 < 32; a3++) {
                __half in1 = x0[a0 * 2048 + a1 * 32 + a3];
                __half in2 = x1[a0 * 2048 + a2 * 32 + a3];
                __half v0 = in1 * in2;
                acc0 += __half2float(v0);
            }
            float v1 = acc0 * __half2float(in0);
            __half in3 = x3[a1 * 64 + a2];
            float v2 = __half2float(in3) + v1;
            float one = 1.0f;
            float v2__gn = fmaxf(acc1, v2);
            float v2__alpha = expf((acc1 - v2__gn));
            float v2__beta = expf((v2 - v2__gn));
            float v2__acc3_sa = acc3 * v2__alpha;
            float v2__acc3_sb = one * v2__beta;
            acc3 = v2__acc3_sa + v2__acc3_sb;
            float v2__acc5__sum_sa = acc5__sum * v2__alpha;
            __half in9 = x2[a0 * 2048 + a2 * 32 + a6];
            float v2__acc5__sum_sb = __half2float(in9) * v2__beta;
            acc5__sum = v2__acc5__sum_sa + v2__acc5__sum_sb;
            acc1 = fmaxf(acc1, v2);
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 16);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 16);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 8);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 8);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 4);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 4);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 2);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 2);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            float acc3__o = __shfl_xor_sync(__activemask(), acc3, 1);
            float acc5__sum__o = __shfl_xor_sync(__activemask(), acc5__sum, 1);
            float acc1__o__gn = fmaxf(acc1, acc1__o);
            float acc1__o__dg = acc1 - acc1__o__gn;
            float acc1__o__alpha = expf(acc1__o__dg);
            float acc1__o__dg_o = acc1__o - acc1__o__gn;
            float acc1__o__beta = expf(acc1__o__dg_o);
            float acc1__o__acc3_sa = acc1__o__alpha * acc3;
            float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
            acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
            float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
            float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
            acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
            acc1 = acc1__o__gn;
        }
        __shared__ float acc1_smem[4];
        __shared__ float acc3_smem[4];
        __shared__ float acc5__sum_smem[4];
        if (lane == 0) {
            acc1_smem[warp] = acc1;
            acc3_smem[warp] = acc3;
            acc5__sum_smem[warp] = acc5__sum;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                float acc3 = acc3_smem[warp];
                float acc3__o = acc3_smem[warp + s];
                float acc5__sum = acc5__sum_smem[warp];
                float acc5__sum__o = acc5__sum_smem[warp + s];
                float acc1__o__gn = fmaxf(acc1, acc1__o);
                float acc1__o__dg = acc1 - acc1__o__gn;
                float acc1__o__alpha = expf(acc1__o__dg);
                float acc1__o__dg_o = acc1__o - acc1__o__gn;
                float acc1__o__beta = expf(acc1__o__dg_o);
                float acc1__o__acc3_sa = acc1__o__alpha * acc3;
                float acc1__o__acc3_sb = acc1__o__beta * acc3__o;
                acc3 = acc1__o__acc3_sa + acc1__o__acc3_sb;
                float acc1__o__acc5__sum_sa = acc1__o__alpha * acc5__sum;
                float acc1__o__acc5__sum_sb = acc1__o__beta * acc5__sum__o;
                acc5__sum = acc1__o__acc5__sum_sa + acc1__o__acc5__sum_sb;
                acc1 = acc1__o__gn;
                acc1_smem[warp] = acc1;
                acc3_smem[warp] = acc3;
                acc5__sum_smem[warp] = acc5__sum;
            }
            __syncthreads();
        }
        acc1 = acc1_smem[0];
        acc3 = acc3_smem[0];
        acc5__sum = acc5__sum_smem[0];
        if (a2_co == 0) {
            float acc5 = acc5__sum * (1.0f / acc3);
            scaled_dot_product_attention[a0 * 2048 + a1 * 32 + a6] = __float2half(acc5);
        }
}
