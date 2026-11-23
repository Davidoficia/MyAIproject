"""
Camada de compatibilidade do motor de sentido (encoder com atenção).
Mesma definição, duas implementações:
  - numpy: só passagem para frente (inferência no sandbox e, depois, no celular)
  - torch: mesma matemática, para o treino no CI
Pesos: dicionário com nomes fixos, salvo em .npz (mesmo formato nos dois lados).

Uso:
  python3 compat_sentido.py paridade   -> compara torch x numpy (precisa de torch, roda no CI)
"""
import math
import numpy as np

PAD = 0
EPS = 1e-5


def init(rng, V, D=64, N=60, L=2, H=4, F=128):
    assert D % H == 0
    p = {"E": rng.normal(0, 0.02, (V, D)), "P": rng.normal(0, 0.02, (N, D))}
    p["E"][PAD] = 0
    for l in range(L):
        for nome in ("Wq", "Wk", "Wv", "Wo"):
            p[f"{nome}{l}"] = rng.normal(0, 1 / math.sqrt(D), (D, D))
        p[f"ln1g{l}"] = np.ones(D); p[f"ln1b{l}"] = np.zeros(D)
        p[f"W1_{l}"] = rng.normal(0, 1 / math.sqrt(D), (D, F)); p[f"b1_{l}"] = np.zeros(F)
        p[f"W2_{l}"] = rng.normal(0, 1 / math.sqrt(F), (F, D)); p[f"b2_{l}"] = np.zeros(D)
        p[f"ln2g{l}"] = np.ones(D); p[f"ln2b{l}"] = np.zeros(D)
    p["lnfg"] = np.ones(D); p["lnfb"] = np.zeros(D)
    return {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}


def salva(caminho, p):
    np.savez(caminho, **p)


def carrega(caminho):
    z = np.load(caminho)
    return {k: z[k] for k in z.files}


# ---------------- numpy ----------------
def _ln_np(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + EPS) * g + b


def _gelu_np(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _softmax_np(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def forward_np(p, ids, H=4):
    B, N = ids.shape
    D = p["E"].shape[1]; dh = D // H; L = sum(1 for k in p if k.startswith("Wq"))
    mask = ids != PAD
    x = p["E"][ids] + p["P"][:N]
    for l in range(L):
        h = _ln_np(x, p[f"ln1g{l}"], p[f"ln1b{l}"])
        q = (h @ p[f"Wq{l}"]).reshape(B, N, H, dh).transpose(0, 2, 1, 3)
        k = (h @ p[f"Wk{l}"]).reshape(B, N, H, dh).transpose(0, 2, 1, 3)
        v = (h @ p[f"Wv{l}"]).reshape(B, N, H, dh).transpose(0, 2, 1, 3)
        s = q @ k.swapaxes(-1, -2) / math.sqrt(dh)
        s = np.where(mask[:, None, None, :], s, -1e9)
        o = (_softmax_np(s) @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        x = x + o @ p[f"Wo{l}"]
        h = _ln_np(x, p[f"ln2g{l}"], p[f"ln2b{l}"])
        x = x + _gelu_np(h @ p[f"W1_{l}"] + p[f"b1_{l}"]) @ p[f"W2_{l}"] + p[f"b2_{l}"]
    x = _ln_np(x, p["lnfg"], p["lnfb"])
    m = mask[..., None].astype(x.dtype)
    return (x * m).sum(1) / np.maximum(m.sum(1), 1)


# ---------------- torch (treino) ----------------
def forward_torch(tp, ids, H=4):
    import torch
    import torch.nn.functional as F_
    B, N = ids.shape
    D = tp["E"].shape[1]; dh = D // H; L = sum(1 for k in tp if k.startswith("Wq"))
    mask = ids != PAD
    x = tp["E"][ids] + tp["P"][:N]
    for l in range(L):
        h = F_.layer_norm(x, (D,), tp[f"ln1g{l}"], tp[f"ln1b{l}"], EPS)
        q = (h @ tp[f"Wq{l}"]).reshape(B, N, H, dh).transpose(1, 2)
        k = (h @ tp[f"Wk{l}"]).reshape(B, N, H, dh).transpose(1, 2)
        v = (h @ tp[f"Wv{l}"]).reshape(B, N, H, dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(dh)
        s = s.masked_fill(~mask[:, None, None, :], -1e9)
        o = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, N, D)
        x = x + o @ tp[f"Wo{l}"]
        h = F_.layer_norm(x, (D,), tp[f"ln2g{l}"], tp[f"ln2b{l}"], EPS)
        u = h @ tp[f"W1_{l}"] + tp[f"b1_{l}"]
        g = 0.5 * u * (1 + torch.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u ** 3)))
        x = x + g @ tp[f"W2_{l}"] + tp[f"b2_{l}"]
    x = F_.layer_norm(x, (D,), tp["lnfg"], tp["lnfb"], EPS)
    m = mask[..., None].to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


def paridade(tol=1e-10):
    import torch
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    p = init(rng, V=500, D=64, N=20, L=2, H=4, F=128)
    ids = rng.integers(3, 500, (8, 20))
    ids[:, 15:] = PAD; ids[3, 10:] = PAD
    a = forward_np(p, ids)
    tp = {k: torch.from_numpy(v) for k, v in p.items()}
    b = forward_torch(tp, torch.from_numpy(ids)).numpy()
    dif = float(np.abs(a - b).max())
    print("max dif numpy x torch:", dif, flush=True)
    assert dif < tol, f"paridade falhou: {dif}"
    print("PARIDADE OK", flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["paridade"]:
        paridade()
