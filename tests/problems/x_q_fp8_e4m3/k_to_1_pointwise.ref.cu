#include <cuda_fp8.h>
extern "C" __global__
__launch_bounds__(256) void k_to_1_pointwise(const float* x, float* to_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float4 _v_in0__u0 = *reinterpret_cast<const float4*>(&x[a0 * 4]);
        float in0__u0 = _v_in0__u0.x;
        float in0__u1 = _v_in0__u0.y;
        float in0__u2 = _v_in0__u0.z;
        float in0__u3 = _v_in0__u0.w;
        __nv_fp8_e4m3 v0__u0 = __nv_fp8_e4m3(in0__u0);
        __nv_fp8_e4m3 v0__u1 = __nv_fp8_e4m3(in0__u1);
        __nv_fp8_e4m3 v0__u2 = __nv_fp8_e4m3(in0__u2);
        __nv_fp8_e4m3 v0__u3 = __nv_fp8_e4m3(in0__u3);
        float4 _vs_v0__u0_72ffd27c = make_float4(float(v0__u0), float(v0__u1), float(v0__u2), float(v0__u3));
        *reinterpret_cast<float4*>(&to_1[a0 * 4]) = _vs_v0__u0_72ffd27c;
}
