"""Shape sweep: hidden-set-like problems at sizes the open set does not cover.

Not part of the regular suite (tests/sweep is gitignored; rebuild it with
`tools/generate.py tests/expressions_sweep.py --out tests/sweep`, run it with
`py tests/run_all.py --sweep`). Each case asks
emmy to pick a schedule for a shape we have not seen, so a lowering gap shows up
here before the hidden set finds it. Same conventions as expressions_extra.py.
"""
import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"
f32 = lambda a: np.asarray(a, dtype=np.float32)
f64 = lambda a: np.asarray(a, dtype=np.float64)
MM = dict(rtol=1e-3, atol=1e-3)
RED = dict(rtol=1e-3, atol=1e-3)


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _mm(m, k, n, **kw):
    return dict(id=f"s_m_{m}x{k}x{n}", code=f"torch.randn({m}, {k}) @ torch.randn({k}, {n})",
                ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM, **kw)


def _attn(s, d, t=None):
    t = t or s
    sc = 1.0 / np.sqrt(d)
    return dict(id=f"s_a_{s}x{t}x{d}",
                code=f"q = torch.randn({s}, {d}); kt = torch.randn({d}, {t}); v = torch.randn({t}, {d}); "
                     f"torch.softmax(q @ kt * {float(sc)!r}, dim=-1) @ v",
                ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) * sc) @ f64(v))], rtol=1e-3, atol=1e-4)


OPEN = [
    # matmul
    _mm(1024, 1024, 1024), _mm(128, 2048, 128), _mm(1, 512, 512), _mm(512, 512, 1),
    _mm(33, 65, 17), _mm(1000, 1000, 1000), _mm(64, 64, 4096), _mm(4096, 64, 64),
    _mm(256, 4096, 256), _mm(768, 768, 768),
    dict(id="s_m_batched_8", code="torch.randn(8, 128, 64) @ torch.randn(8, 64, 128)",
         ref=lambda a, b: [f32(f64(a) @ f64(b))], **MM),
    # attention
    _attn(512, 64), _attn(1024, 64), _attn(100, 32), _attn(256, 128), _attn(16, 8), _attn(64, 64, 1000),
    dict(id="s_a_heads8", code="q = torch.randn(8, 128, 64); kt = torch.randn(8, 64, 128); v = torch.randn(8, 128, 64); "
                              "torch.softmax(q @ kt * 0.125, dim=-1) @ v",
         ref=lambda q, kt, v: [f32(_softmax((f64(q) @ f64(kt)) * 0.125) @ f64(v))], rtol=1e-3, atol=1e-4),
    # reductions, pinned like the open set and with emmy's default schedule
    *[dict(id=f"s_r_{name}{tag}", code=code, ref=ref, **RED, **kn)
      for tag, kn in (("", dict(knobs=REDUCE_KNOBS)), ("_dflt", {}))
      for name, code, ref in (
          ("sum_1m", "torch.randn(1048576).sum()", lambda x: [f32(f64(x).sum())]),
          ("sum_prime", "torch.randn(1000003).sum()", lambda x: [f32(f64(x).sum())]),
          ("mean_rows", "torch.randn(1000, 1000).mean(-1)", lambda x: [f32(f64(x).mean(-1))]),
          ("max_rows", "torch.randn(512, 2048).amax(-1)", lambda x: [f32(x.max(-1))]),
          ("sum_dim0", "torch.randn(1024, 1024).sum(0)", lambda x: [f32(f64(x).sum(0))]),
          ("sum_4d", "torch.randn(4, 8, 16, 256).sum(-1)", lambda x: [f32(f64(x).sum(-1))]),
          ("sumsq_rows", "x = torch.randn(256, 3000); (x * x).sum(-1)", lambda x: [f32((f64(x) ** 2).sum(-1))]),
      )],
    # fused
    dict(id="s_f_softmax_big", code="torch.softmax(torch.randn(1024, 4096), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-6),
    dict(id="s_f_layernorm_big", code="x = torch.randn(512, 1024); "
         "(x - x.mean(-1, keepdim=True)) / torch.sqrt(((x - x.mean(-1, keepdim=True)) ** 2).mean(-1, keepdim=True) + 1e-5)",
         ref=lambda x: [f32((f64(x) - f64(x).mean(-1, keepdims=True)) / np.sqrt(f64(x).var(-1, keepdims=True) + 1e-5))],
         rtol=1e-3, atol=1e-4),
    dict(id="s_f_rmsnorm_big", code="x = torch.randn(1024, 2048); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)",
         ref=lambda x: [f32(f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-6))], rtol=1e-3, atol=1e-4),
    dict(id="s_f_l2norm_odd", code="x = torch.randn(77, 999); x / torch.sqrt((x * x).sum(-1, keepdim=True))",
         ref=lambda x: [f32(f64(x) / np.sqrt((f64(x) ** 2).sum(-1, keepdims=True)))], rtol=1e-3, atol=1e-5),
    dict(id="s_f_softmax_3d", code="torch.softmax(torch.randn(8, 64, 256), dim=-1)",
         ref=lambda x: [f32(_softmax(f64(x)))], rtol=1e-3, atol=1e-6),
    dict(id="s_f_logsoftmax_big", code="x = torch.randn(256, 2048); "
         "x - x.amax(-1, keepdim=True) - torch.log(torch.exp(x - x.amax(-1, keepdim=True)).sum(-1, keepdim=True))",
         ref=lambda x: [f32((lambda y: y - np.log(np.exp(y).sum(-1, keepdims=True)))(f64(x) - f64(x).max(-1, keepdims=True)))],
         rtol=1e-4, atol=1e-4),
    dict(id="s_f_mm_silu", code="torch.nn.functional.silu(torch.randn(256, 256) @ torch.randn(256, 512))",
         ref=lambda a, b: [f32((lambda y: y / (1 + np.exp(-y)))(f64(a) @ f64(b)))], **MM),
    # pointwise (the runner rejects inputs of 8M+ elements with HTTP 422)
    dict(id="s_p_2m", code="torch.randn(2097152) * 2.0 + 1.0",
         ref=lambda x: [x * np.float32(2) + np.float32(1)], rtol=0, atol=0),
    dict(id="s_p_4d_bcast", code="torch.randn(2, 3, 64, 128) * torch.randn(1, 3, 1, 128) + torch.randn(128)",
         ref=lambda a, b, c: [a * b + c], rtol=0, atol=0),
    dict(id="s_p_chain", code="x = torch.randn(65536); torch.tanh(torch.exp(-x * x) * torch.sigmoid(x) + torch.sqrt(torch.abs(x)))",
         ref=lambda x: [f32(np.tanh(np.exp(-f64(x) ** 2) / (1 + np.exp(-f64(x))) + np.sqrt(np.abs(f64(x)))))],
         rtol=1e-5, atol=1e-6),
    dict(id="s_p_scalar_tensor", code="torch.randn(1) * torch.randn(4096)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
]
