"""Modern-LLM building blocks, as emmy traces them: SSM scans, linear attention,
SwiGLU, RoPE, GQA, softcapping, RMSNorm. Same format as expressions_extra.py.

Inputs are seeded with standard normals, so a positive or decaying input is made
inside the expression (abs, scale) and the reference repeats that step.
"""

import numpy as np

f64 = lambda a: np.asarray(a).astype(np.float64)
f32 = lambda a: np.asarray(a).astype(np.float32)
f16 = lambda a: np.asarray(a).astype(np.float16)
H = "dtype=torch.float16"
F = dict(rtol=1e-4, atol=1e-4)
MM = dict(rtol=1e-3, atol=1e-3)


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _silu(x):
    return x / (1 + np.exp(-x))


def _decay(x):
    """The SSM log-decay: negative, small, so exp(cumsum) stays in range."""
    return -np.abs(f64(x)) * 0.1


def _ssm(x, u):
    """Mamba-2 style scan in matrix form: y_t = sum_{s<=t} exp(cs_t - cs_s) u_s."""
    cs = np.cumsum(_decay(x))
    return f32(np.tril(np.exp(cs[:, None] - cs[None, :])) @ f64(u))


def _linattn(q, k, v):
    """Causal linear attention: y_t = q_t . sum_{s<=t} k_s v_s^T."""
    state = np.cumsum(f64(k)[:, :, None] * f64(v)[:, None, :], 0)
    return f32((f64(q)[:, :, None] * state).sum(1))


def _rope(x, c, s):
    x = f64(x)
    return f32(x * c + np.concatenate((-x[:, 32:], x[:, :32]), -1) * s)


def _attn(q, k, v):
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    return _softmax(s) @ v


OPEN = [
    # ---- state-space models (Mamba): decays, scans ----
    dict(id="x_l_ssm_decay", code="(lambda a, x: torch.exp(torch.cumsum(a, -1)) * x)"
                                   "(-torch.randn(16, 256).abs() * 0.1, torch.randn(16, 256))",
         ref=lambda a, x: [f32(np.exp(np.cumsum(_decay(a), -1)) * f64(x))], **F),
    dict(id="x_l_ssm_scan", code="(lambda a, u: torch.tril(torch.exp(torch.cumsum(a, 0)[:, None] "
                                  "- torch.cumsum(a, 0)[None, :])) @ u)(-torch.randn(128).abs() * 0.1, torch.randn(128, 16))",
         ref=lambda a, u: [_ssm(a, u)], **MM),
    dict(id="x_l_linattn", code="(lambda q, k, v: (q[:, :, None] * torch.cumsum(k[:, :, None] * v[:, None, :], 0)).sum(1))"
                                "(torch.randn(64, 16), torch.randn(64, 16), torch.randn(64, 16))",
         ref=lambda q, k, v: [_linattn(q, k, v)], **MM),
    # ---- transformer blocks ----
    dict(id="x_l_swiglu", code="(lambda x, w1, w3: torch.nn.functional.silu(x @ w1) * (x @ w3))"
                               "(torch.randn(64, 128), torch.randn(128, 256), torch.randn(128, 256))",
         ref=lambda x, w1, w3: [f32(_silu(f64(x) @ f64(w1)) * (f64(x) @ f64(w3)))], **MM),
    dict(id="x_l_swiglu_h", code=f"(lambda x, w1, w3: torch.nn.functional.silu(x @ w1) * (x @ w3))"
                                 f"(torch.randn(64, 128, {H}), torch.randn(128, 256, {H}), torch.randn(128, 256, {H}))",
         ref=lambda x, w1, w3: [f16(_silu(f64(f16(f64(x) @ f64(w1)))) * f64(f16(f64(x) @ f64(w3))))],
         rtol=1e-2, atol=1e-2),
    dict(id="x_l_rope", code="(lambda x, c, s: x * c + torch.cat((-x[:, 32:], x[:, :32]), -1) * s)"
                             "(torch.randn(128, 64), torch.randn(128, 64), torch.randn(128, 64))",
         ref=lambda x, c, s: [_rope(x, c, s)], **F),
    dict(id="x_l_rmsnorm_w", code="(lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w)"
                                  "(torch.randn(64, 512), torch.randn(512))",
         ref=lambda x, w: [f32(f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-6) * f64(w))], **F),
    dict(id="x_l_softcap", code="torch.softmax(30.0 * torch.tanh(torch.randn(64, 256) / 30.0), -1)",
         ref=lambda x: [f32(_softmax(30 * np.tanh(f64(x) / 30)))], **F),
    dict(id="x_l_gqa", code="(lambda q, k, v: torch.nn.functional.scaled_dot_product_attention("
                            "q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)))"
                            "(torch.randn(1, 8, 64, 32), torch.randn(1, 2, 64, 32), torch.randn(1, 2, 64, 32))",
         ref=lambda q, k, v: [f32(_attn(f64(q), np.repeat(f64(k), 4, 1), np.repeat(f64(v), 4, 1)))], **MM),
]
