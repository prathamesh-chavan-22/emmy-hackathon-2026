"""Low-precision formats: bf16, fp8 (e4m3, e5m2), int4. The runner only
allocates f32/f16/f64/i32/i64 buffers, so these appear the way inference uses
them: quantize/dequantize inside a kernel, or packed into i32 words. int8 and
fp4 cannot be tested: emmy 0.3.11 has no int8 dtype (tracing .to(torch.int8)
fails), and its fp4 (f4e2m1x2) has no torch cast to reach it.
References use torch itself (this file runs in the generator venv).
"""

import torch

t = torch.from_numpy
I32P = "torch.randint(-2**31, 2**31 - 1, (256, 64), dtype=torch.int32)"


OPEN = [
    dict(id="x_q_fp8_e4m3", code="torch.randn(4096).to(torch.float8_e4m3fn).float()",
         ref=lambda x: [t(x).to(torch.float8_e4m3fn).float().numpy()], rtol=0, atol=0),
    dict(id="x_q_fp8_e5m2", code="torch.randn(4096).to(torch.float8_e5m2).float()",
         ref=lambda x: [t(x).to(torch.float8_e5m2).float().numpy()], rtol=0, atol=0),
    # Over the range: torch saturates e4m3fn to +-448 and takes e5m2 to +-inf.
    dict(id="x_q_fp8_e4m3_sat", code="(torch.randn(4096) * 300.0).to(torch.float8_e4m3fn).float()",
         ref=lambda x: [(t(x) * 300.0).to(torch.float8_e4m3fn).float().numpy()], rtol=0, atol=0),
    # The runner's JSON returns inf as null, so the infinities are clamped after the cast.
    dict(id="x_q_fp8_e5m2_inf", code="(torch.randn(4096) * 40000.0).to(torch.float8_e5m2).float().clamp(-1e5, 1e5)",
         ref=lambda x: [(t(x) * 40000.0).to(torch.float8_e5m2).float().clamp(-1e5, 1e5).numpy()], rtol=0, atol=0),
    dict(id="x_q_int4_unpack", code=f"((({I32P} >> 4) & 15) - 8).float() * 0.1",
         ref=lambda p: [((((t(p) >> 4) & 15) - 8).float() * 0.1).numpy()], rtol=0, atol=0),
]
OPEN += [
    dict(id="x_q_bf16_cast", code="torch.randn(4096).to(torch.bfloat16).float()",
         ref=lambda x: [t(x).to(torch.bfloat16).float().numpy()], rtol=0, atol=0),
    dict(id="x_q_bf16_math", code="(lambda a, b: (a.to(torch.bfloat16) * b.to(torch.bfloat16) + a.to(torch.bfloat16)).float())"
                                  "(torch.randn(4096), torch.randn(4096))",
         ref=lambda a, b: [(t(a).bfloat16() * t(b).bfloat16() + t(a).bfloat16()).float().numpy()], rtol=0, atol=0),
    dict(id="x_q_fp8_scaled", code="(lambda x: (x * 64.0).to(torch.float8_e4m3fn).float() / 64.0)(torch.randn(64, 256))",
         ref=lambda x: [((t(x) * 64.0).to(torch.float8_e4m3fn).float() / 64.0).numpy()], rtol=0, atol=0),
    # emmy's IR stores the f32 accumulator; torch rounds the product to bf16 before .float(),
    # and that step is not in the IR. So this one matches to half a bf16 ulp, not bit for bit.
    dict(id="x_q_bf16_mm", code="(lambda a, b: (a.to(torch.bfloat16) @ b.to(torch.bfloat16)).float())"
                                "(torch.randn(64, 64), torch.randn(64, 64))",
         ref=lambda a, b: [(t(a).bfloat16() @ t(b).bfloat16()).float().numpy()], rtol=1e-2, atol=1e-2),
    # A Python scalar stays f32 in torch even next to a half tensor: f16(float(x) * 0.1f).
    dict(id="x_q_f16_scalar", code="(torch.randn(4096).half() * 0.1).float()",
         ref=lambda x: [(t(x).half() * 0.1).float().numpy()], rtol=0, atol=0),
]
