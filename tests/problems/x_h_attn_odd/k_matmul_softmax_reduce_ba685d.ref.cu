#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul_softmax_reduce_ba685d(const float* matmul_1__place_03e2a34518_0, const __half* div_c1, const __half* v, __half* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 156) {
        int a0_b = _gid / 12;
        int a5_b = (_gid) % 12;
        int a0_u = (_gid) % 1;
        int a5_u = (_gid) % 1;
        __half in0 = div_c1[0];
        __half v2 = __float2half(1.0f) / in0;
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        __half acc1__c0_0 = __float2half(-1e+30f);
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        __half acc1__c0_1 = __float2half(-1e+30f);
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        __half acc1__c1_0 = __float2half(-1e+30f);
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        __half acc1__c1_1 = __float2half(-1e+30f);
        float acc3__c2_0 = 0.0f;
        float acc5__sum__c2_0 = 0.0f;
        __half acc1__c2_0 = __float2half(-1e+30f);
        float acc3__c2_1 = 0.0f;
        float acc5__sum__c2_1 = 0.0f;
        __half acc1__c2_1 = __float2half(-1e+30f);
        float acc3__c3_0 = 0.0f;
        float acc5__sum__c3_0 = 0.0f;
        __half acc1__c3_0 = __float2half(-1e+30f);
        float acc3__c3_1 = 0.0f;
        float acc5__sum__c3_1 = 0.0f;
        __half acc1__c3_1 = __float2half(-1e+30f);
        for (int a1 = 0; a1 < 70; a1++) {
            float acc0__ws03e2a34518__ar0 = matmul_1__place_03e2a34518_0[(a0_b * 4 + a0_u * 4) % 50 * 70 + a1];
            __half v4__ws03e2a34518__c0_0 = __float2half(acc0__ws03e2a34518__ar0);
            __half v5__ws03e2a34518__c0_0 = v2 * v4__ws03e2a34518__c0_0;
            float one__c0_0 = 1.0f;
            float v5__ws03e2a34518__gn__c0_0 = fmaxf(__half2float(acc1__c0_0), __half2float(v5__ws03e2a34518__c0_0));
            float v5__ws03e2a34518__dg__c0_0 = __half2float(acc1__c0_0) - v5__ws03e2a34518__gn__c0_0;
            float v5__ws03e2a34518__alpha__c0_0 = expf(v5__ws03e2a34518__dg__c0_0);
            float v5__ws03e2a34518__dg_o__c0_0 = __half2float(v5__ws03e2a34518__c0_0) - v5__ws03e2a34518__gn__c0_0;
            float v5__ws03e2a34518__beta__c0_0 = expf(v5__ws03e2a34518__dg_o__c0_0);
            float v5__ws03e2a34518__acc3_sa__c0_0 = acc3__c0_0 * v5__ws03e2a34518__alpha__c0_0;
            float v5__ws03e2a34518__acc3_sb__c0_0 = one__c0_0 * v5__ws03e2a34518__beta__c0_0;
            acc3__c0_0 = v5__ws03e2a34518__acc3_sa__c0_0 + v5__ws03e2a34518__acc3_sb__c0_0;
            float v5__ws03e2a34518__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v5__ws03e2a34518__alpha__c0_0;
            __half2 _v_in7__bc0 = *reinterpret_cast<const __half2*>(&v[a1 * 24 + (a5_b * 2 + a5_u * 2)]);
            __half in7__bc0 = _v_in7__bc0.x;
            __half in7__bc1 = _v_in7__bc0.y;
            float v5__ws03e2a34518__acc5__sum_sb__c0_0 = __half2float(in7__bc0) * v5__ws03e2a34518__beta__c0_0;
            acc5__sum__c0_0 = v5__ws03e2a34518__acc5__sum_sa__c0_0 + v5__ws03e2a34518__acc5__sum_sb__c0_0;
            acc1__c0_0 = __hmax(acc1__c0_0, v5__ws03e2a34518__c0_0);
            __half v4__ws03e2a34518__c0_1 = __float2half(acc0__ws03e2a34518__ar0);
            __half v5__ws03e2a34518__c0_1 = v2 * v4__ws03e2a34518__c0_1;
            float one__c0_1 = 1.0f;
            float v5__ws03e2a34518__gn__c0_1 = fmaxf(__half2float(acc1__c0_1), __half2float(v5__ws03e2a34518__c0_1));
            float v5__ws03e2a34518__dg__c0_1 = __half2float(acc1__c0_1) - v5__ws03e2a34518__gn__c0_1;
            float v5__ws03e2a34518__alpha__c0_1 = expf(v5__ws03e2a34518__dg__c0_1);
            float v5__ws03e2a34518__dg_o__c0_1 = __half2float(v5__ws03e2a34518__c0_1) - v5__ws03e2a34518__gn__c0_1;
            float v5__ws03e2a34518__beta__c0_1 = expf(v5__ws03e2a34518__dg_o__c0_1);
            float v5__ws03e2a34518__acc3_sa__c0_1 = acc3__c0_1 * v5__ws03e2a34518__alpha__c0_1;
            float v5__ws03e2a34518__acc3_sb__c0_1 = one__c0_1 * v5__ws03e2a34518__beta__c0_1;
            acc3__c0_1 = v5__ws03e2a34518__acc3_sa__c0_1 + v5__ws03e2a34518__acc3_sb__c0_1;
            float v5__ws03e2a34518__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v5__ws03e2a34518__alpha__c0_1;
            float v5__ws03e2a34518__acc5__sum_sb__c0_1 = __half2float(in7__bc1) * v5__ws03e2a34518__beta__c0_1;
            acc5__sum__c0_1 = v5__ws03e2a34518__acc5__sum_sa__c0_1 + v5__ws03e2a34518__acc5__sum_sb__c0_1;
            acc1__c0_1 = __hmax(acc1__c0_1, v5__ws03e2a34518__c0_1);
            float acc0__ws03e2a34518__ar1 = matmul_1__place_03e2a34518_0[(a0_b * 4 + a0_u * 4 + 1) % 50 * 70 + a1];
            __half v4__ws03e2a34518__c1_0 = __float2half(acc0__ws03e2a34518__ar1);
            __half v5__ws03e2a34518__c1_0 = v2 * v4__ws03e2a34518__c1_0;
            float one__c1_0 = 1.0f;
            float v5__ws03e2a34518__gn__c1_0 = fmaxf(__half2float(acc1__c1_0), __half2float(v5__ws03e2a34518__c1_0));
            float v5__ws03e2a34518__dg__c1_0 = __half2float(acc1__c1_0) - v5__ws03e2a34518__gn__c1_0;
            float v5__ws03e2a34518__alpha__c1_0 = expf(v5__ws03e2a34518__dg__c1_0);
            float v5__ws03e2a34518__dg_o__c1_0 = __half2float(v5__ws03e2a34518__c1_0) - v5__ws03e2a34518__gn__c1_0;
            float v5__ws03e2a34518__beta__c1_0 = expf(v5__ws03e2a34518__dg_o__c1_0);
            float v5__ws03e2a34518__acc3_sa__c1_0 = acc3__c1_0 * v5__ws03e2a34518__alpha__c1_0;
            float v5__ws03e2a34518__acc3_sb__c1_0 = one__c1_0 * v5__ws03e2a34518__beta__c1_0;
            acc3__c1_0 = v5__ws03e2a34518__acc3_sa__c1_0 + v5__ws03e2a34518__acc3_sb__c1_0;
            float v5__ws03e2a34518__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v5__ws03e2a34518__alpha__c1_0;
            float v5__ws03e2a34518__acc5__sum_sb__c1_0 = __half2float(in7__bc0) * v5__ws03e2a34518__beta__c1_0;
            acc5__sum__c1_0 = v5__ws03e2a34518__acc5__sum_sa__c1_0 + v5__ws03e2a34518__acc5__sum_sb__c1_0;
            acc1__c1_0 = __hmax(acc1__c1_0, v5__ws03e2a34518__c1_0);
            __half v4__ws03e2a34518__c1_1 = __float2half(acc0__ws03e2a34518__ar1);
            __half v5__ws03e2a34518__c1_1 = v2 * v4__ws03e2a34518__c1_1;
            float one__c1_1 = 1.0f;
            float v5__ws03e2a34518__gn__c1_1 = fmaxf(__half2float(acc1__c1_1), __half2float(v5__ws03e2a34518__c1_1));
            float v5__ws03e2a34518__dg__c1_1 = __half2float(acc1__c1_1) - v5__ws03e2a34518__gn__c1_1;
            float v5__ws03e2a34518__alpha__c1_1 = expf(v5__ws03e2a34518__dg__c1_1);
            float v5__ws03e2a34518__dg_o__c1_1 = __half2float(v5__ws03e2a34518__c1_1) - v5__ws03e2a34518__gn__c1_1;
            float v5__ws03e2a34518__beta__c1_1 = expf(v5__ws03e2a34518__dg_o__c1_1);
            float v5__ws03e2a34518__acc3_sa__c1_1 = acc3__c1_1 * v5__ws03e2a34518__alpha__c1_1;
            float v5__ws03e2a34518__acc3_sb__c1_1 = one__c1_1 * v5__ws03e2a34518__beta__c1_1;
            acc3__c1_1 = v5__ws03e2a34518__acc3_sa__c1_1 + v5__ws03e2a34518__acc3_sb__c1_1;
            float v5__ws03e2a34518__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v5__ws03e2a34518__alpha__c1_1;
            float v5__ws03e2a34518__acc5__sum_sb__c1_1 = __half2float(in7__bc1) * v5__ws03e2a34518__beta__c1_1;
            acc5__sum__c1_1 = v5__ws03e2a34518__acc5__sum_sa__c1_1 + v5__ws03e2a34518__acc5__sum_sb__c1_1;
            acc1__c1_1 = __hmax(acc1__c1_1, v5__ws03e2a34518__c1_1);
            float acc0__ws03e2a34518__ar2 = matmul_1__place_03e2a34518_0[(a0_b * 4 + a0_u * 4 + 2) % 50 * 70 + a1];
            __half v4__ws03e2a34518__c2_0 = __float2half(acc0__ws03e2a34518__ar2);
            __half v5__ws03e2a34518__c2_0 = v2 * v4__ws03e2a34518__c2_0;
            float one__c2_0 = 1.0f;
            float v5__ws03e2a34518__gn__c2_0 = fmaxf(__half2float(acc1__c2_0), __half2float(v5__ws03e2a34518__c2_0));
            float v5__ws03e2a34518__dg__c2_0 = __half2float(acc1__c2_0) - v5__ws03e2a34518__gn__c2_0;
            float v5__ws03e2a34518__alpha__c2_0 = expf(v5__ws03e2a34518__dg__c2_0);
            float v5__ws03e2a34518__dg_o__c2_0 = __half2float(v5__ws03e2a34518__c2_0) - v5__ws03e2a34518__gn__c2_0;
            float v5__ws03e2a34518__beta__c2_0 = expf(v5__ws03e2a34518__dg_o__c2_0);
            float v5__ws03e2a34518__acc3_sa__c2_0 = acc3__c2_0 * v5__ws03e2a34518__alpha__c2_0;
            float v5__ws03e2a34518__acc3_sb__c2_0 = one__c2_0 * v5__ws03e2a34518__beta__c2_0;
            acc3__c2_0 = v5__ws03e2a34518__acc3_sa__c2_0 + v5__ws03e2a34518__acc3_sb__c2_0;
            float v5__ws03e2a34518__acc5__sum_sa__c2_0 = acc5__sum__c2_0 * v5__ws03e2a34518__alpha__c2_0;
            float v5__ws03e2a34518__acc5__sum_sb__c2_0 = __half2float(in7__bc0) * v5__ws03e2a34518__beta__c2_0;
            acc5__sum__c2_0 = v5__ws03e2a34518__acc5__sum_sa__c2_0 + v5__ws03e2a34518__acc5__sum_sb__c2_0;
            acc1__c2_0 = __hmax(acc1__c2_0, v5__ws03e2a34518__c2_0);
            __half v4__ws03e2a34518__c2_1 = __float2half(acc0__ws03e2a34518__ar2);
            __half v5__ws03e2a34518__c2_1 = v2 * v4__ws03e2a34518__c2_1;
            float one__c2_1 = 1.0f;
            float v5__ws03e2a34518__gn__c2_1 = fmaxf(__half2float(acc1__c2_1), __half2float(v5__ws03e2a34518__c2_1));
            float v5__ws03e2a34518__dg__c2_1 = __half2float(acc1__c2_1) - v5__ws03e2a34518__gn__c2_1;
            float v5__ws03e2a34518__alpha__c2_1 = expf(v5__ws03e2a34518__dg__c2_1);
            float v5__ws03e2a34518__dg_o__c2_1 = __half2float(v5__ws03e2a34518__c2_1) - v5__ws03e2a34518__gn__c2_1;
            float v5__ws03e2a34518__beta__c2_1 = expf(v5__ws03e2a34518__dg_o__c2_1);
            float v5__ws03e2a34518__acc3_sa__c2_1 = acc3__c2_1 * v5__ws03e2a34518__alpha__c2_1;
            float v5__ws03e2a34518__acc3_sb__c2_1 = one__c2_1 * v5__ws03e2a34518__beta__c2_1;
            acc3__c2_1 = v5__ws03e2a34518__acc3_sa__c2_1 + v5__ws03e2a34518__acc3_sb__c2_1;
            float v5__ws03e2a34518__acc5__sum_sa__c2_1 = acc5__sum__c2_1 * v5__ws03e2a34518__alpha__c2_1;
            float v5__ws03e2a34518__acc5__sum_sb__c2_1 = __half2float(in7__bc1) * v5__ws03e2a34518__beta__c2_1;
            acc5__sum__c2_1 = v5__ws03e2a34518__acc5__sum_sa__c2_1 + v5__ws03e2a34518__acc5__sum_sb__c2_1;
            acc1__c2_1 = __hmax(acc1__c2_1, v5__ws03e2a34518__c2_1);
            float acc0__ws03e2a34518__ar3 = matmul_1__place_03e2a34518_0[(a0_b * 4 + a0_u * 4 + 3) % 50 * 70 + a1];
            __half v4__ws03e2a34518__c3_0 = __float2half(acc0__ws03e2a34518__ar3);
            __half v5__ws03e2a34518__c3_0 = v2 * v4__ws03e2a34518__c3_0;
            float one__c3_0 = 1.0f;
            float v5__ws03e2a34518__gn__c3_0 = fmaxf(__half2float(acc1__c3_0), __half2float(v5__ws03e2a34518__c3_0));
            float v5__ws03e2a34518__dg__c3_0 = __half2float(acc1__c3_0) - v5__ws03e2a34518__gn__c3_0;
            float v5__ws03e2a34518__alpha__c3_0 = expf(v5__ws03e2a34518__dg__c3_0);
            float v5__ws03e2a34518__dg_o__c3_0 = __half2float(v5__ws03e2a34518__c3_0) - v5__ws03e2a34518__gn__c3_0;
            float v5__ws03e2a34518__beta__c3_0 = expf(v5__ws03e2a34518__dg_o__c3_0);
            float v5__ws03e2a34518__acc3_sa__c3_0 = acc3__c3_0 * v5__ws03e2a34518__alpha__c3_0;
            float v5__ws03e2a34518__acc3_sb__c3_0 = one__c3_0 * v5__ws03e2a34518__beta__c3_0;
            acc3__c3_0 = v5__ws03e2a34518__acc3_sa__c3_0 + v5__ws03e2a34518__acc3_sb__c3_0;
            float v5__ws03e2a34518__acc5__sum_sa__c3_0 = acc5__sum__c3_0 * v5__ws03e2a34518__alpha__c3_0;
            float v5__ws03e2a34518__acc5__sum_sb__c3_0 = __half2float(in7__bc0) * v5__ws03e2a34518__beta__c3_0;
            acc5__sum__c3_0 = v5__ws03e2a34518__acc5__sum_sa__c3_0 + v5__ws03e2a34518__acc5__sum_sb__c3_0;
            acc1__c3_0 = __hmax(acc1__c3_0, v5__ws03e2a34518__c3_0);
            __half v4__ws03e2a34518__c3_1 = __float2half(acc0__ws03e2a34518__ar3);
            __half v5__ws03e2a34518__c3_1 = v2 * v4__ws03e2a34518__c3_1;
            float one__c3_1 = 1.0f;
            float v5__ws03e2a34518__gn__c3_1 = fmaxf(__half2float(acc1__c3_1), __half2float(v5__ws03e2a34518__c3_1));
            float v5__ws03e2a34518__dg__c3_1 = __half2float(acc1__c3_1) - v5__ws03e2a34518__gn__c3_1;
            float v5__ws03e2a34518__alpha__c3_1 = expf(v5__ws03e2a34518__dg__c3_1);
            float v5__ws03e2a34518__dg_o__c3_1 = __half2float(v5__ws03e2a34518__c3_1) - v5__ws03e2a34518__gn__c3_1;
            float v5__ws03e2a34518__beta__c3_1 = expf(v5__ws03e2a34518__dg_o__c3_1);
            float v5__ws03e2a34518__acc3_sa__c3_1 = acc3__c3_1 * v5__ws03e2a34518__alpha__c3_1;
            float v5__ws03e2a34518__acc3_sb__c3_1 = one__c3_1 * v5__ws03e2a34518__beta__c3_1;
            acc3__c3_1 = v5__ws03e2a34518__acc3_sa__c3_1 + v5__ws03e2a34518__acc3_sb__c3_1;
            float v5__ws03e2a34518__acc5__sum_sa__c3_1 = acc5__sum__c3_1 * v5__ws03e2a34518__alpha__c3_1;
            float v5__ws03e2a34518__acc5__sum_sb__c3_1 = __half2float(in7__bc1) * v5__ws03e2a34518__beta__c3_1;
            acc5__sum__c3_1 = v5__ws03e2a34518__acc5__sum_sa__c3_1 + v5__ws03e2a34518__acc5__sum_sb__c3_1;
            acc1__c3_1 = __hmax(acc1__c3_1, v5__ws03e2a34518__c3_1);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a0_b * 4 + a0_u * 4 + 0 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4) % 50 * 24 + (a5_b * 2 + a5_u * 2)] = __float2half(acc5__c0_0);
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a0_b * 4 + a0_u * 4 + 0 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4) % 50 * 24 + (a5_b * 2 + a5_u * 2 + 1)] = __float2half(acc5__c0_1);
        }
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        if (a0_b * 4 + a0_u * 4 + 1 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) % 50 * 24 + (a5_b * 2 + a5_u * 2)] = __float2half(acc5__c1_0);
        }
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        if (a0_b * 4 + a0_u * 4 + 1 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 1) % 50 * 24 + (a5_b * 2 + a5_u * 2 + 1)] = __float2half(acc5__c1_1);
        }
        float acc5__c2_0 = acc5__sum__c2_0 * (1.0f / acc3__c2_0);
        if (a0_b * 4 + a0_u * 4 + 2 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) % 50 * 24 + (a5_b * 2 + a5_u * 2)] = __float2half(acc5__c2_0);
        }
        float acc5__c2_1 = acc5__sum__c2_1 * (1.0f / acc3__c2_1);
        if (a0_b * 4 + a0_u * 4 + 2 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 2) % 50 * 24 + (a5_b * 2 + a5_u * 2 + 1)] = __float2half(acc5__c2_1);
        }
        float acc5__c3_0 = acc5__sum__c3_0 * (1.0f / acc3__c3_0);
        if (a0_b * 4 + a0_u * 4 + 3 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) % 50 * 24 + (a5_b * 2 + a5_u * 2)] = __float2half(acc5__c3_0);
        }
        float acc5__c3_1 = acc5__sum__c3_1 * (1.0f / acc3__c3_1);
        if (a0_b * 4 + a0_u * 4 + 3 < 50) {
            matmul_1[(a0_b * 4 + a0_u * 4 + 3) % 50 * 24 + (a5_b * 2 + a5_u * 2 + 1)] = __float2half(acc5__c3_1);
        }
    }
}
