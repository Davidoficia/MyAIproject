"""
Junta NM + SM + Nx num escorador só (log-linear), no mesmo teste por artigo do NM v2
(1500 frases, RNG 7, 30% de máscara). Não substitui nenhuma peça.

- NM : contagem skip-gram com estimador "mix" (0.8 contagem + 0.2 unigrama), 2M frases
- SM : rede pequena (2 à esquerda -> próxima), treinada aqui no corpus grande
- Nx : último checkpoint salvo em PESOS_DIR (nx_grande.npz / .json), NX_PASSES passes

Pesos (a, b, c) do NM, Nx e SM são escolhidos na metade par dos casos e medidos na
metade ímpar. Ablação: zera uma peça de cada vez com os pesos escolhidos.

Variáveis de ambiente: DADOS, PESOS_DIR, NTE, NM_FRASES, SM_FRASES, SM_PASSOS, NX_PASSES.
"""
import array, json, os, re, time
from collections import Counter
import numpy as np

DIR = os.environ.get("DADOS", "dados_grande")
PESOS = os.environ.get("PESOS_DIR", "pesos")
NTE = int(os.environ.get("NTE", "1500"))
NM_FRASES = int(os.environ.get("NM_FRASES", "2000000"))
SM_FRASES = int(os.environ.get("SM_FRASES", "2000000"))
SM_PASSOS = int(os.environ.get("SM_PASSOS", "3000"))
NX_PASSES = int(os.environ.get("NX_PASSES", "4"))
L = 32; D = 64; HD = 128; PAD, UNK, MASK = 0, 1, 2
TOK = re.compile(r"\w+|[^\w\s]")
tokens = lambda s: TOK.findall(s.lower())[:L]
D_RNG = np.random.default_rng(0)
t0 = time.time()

# ---------- Nx: vocabulário e pesos do checkpoint (o vocabulário vale pras três peças)
z = np.load(os.path.join(PESOS, "nx_grande.npz"))
meta = json.load(open(os.path.join(PESOS, "nx_grande.json"), encoding="utf-8"))
itos = meta["itos"]; stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
E = z["E"]; uni = z["uni"]
W = {k: z[k] for k in ["W1", "b1", "W2", "b2"]}
print(f"Nx passo {meta['passo']} | V={V} | {time.time()-t0:.0f}s", flush=True)

def carrega(caminho, limite=None):
    flat = array.array("i"); off = array.array("q", [0])
    with open(caminho, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if limite is not None and i >= limite: break
            ids = [stoi.get(w, UNK) for w in tokens(l)]
            if len(ids) < 3: continue
            flat.extend(ids); off.append(len(flat))
    return np.frombuffer(flat, dtype=np.int32).astype(np.int64), np.frombuffer(off, dtype=np.int64).copy()

def monta(Tok, mask, P):
    Lc = Tok.shape[1]
    X = E[Tok].copy()
    X[mask] = P[mask] @ E
    Bn = Tok.shape[0]
    Xp = np.zeros((Bn, Lc + 4, D), np.float32); Xp[:, 2:Lc+2] = X
    F = np.concatenate([Xp[:, 0:Lc], Xp[:, 1:Lc+1], X, Xp[:, 3:Lc+3], Xp[:, 4:Lc+4]], axis=2)
    return F[mask]

def passe(F):
    H = np.tanh(F @ W["W1"] + W["b1"])
    zz = H @ W["W2"] + W["b2"]
    zz -= zz.max(1, keepdims=True); p = np.exp(zz); p /= p.sum(1, keepdims=True)
    return p

# ---------- teste: mesmo sorteio do nm_v2 (RNG 7, uma chamada por frase)
tf, to = carrega(f"{DIR}/teste.txt", NTE * 3)
n_te = min(NTE, len(to) - 1)
r = np.random.default_rng(7)
sents = []
for i in range(n_te):
    s = tf[to[i]:to[i+1]][:L]
    m = r.random(len(s)) < 0.3
    if not m.any(): m[0] = True
    sents.append((s, m))

pos, ys, lref, rref, l2ref = [], [], [], [], []
for i, (s, m) in enumerate(sents):
    cur = [MASK if m[j] else int(s[j]) for j in range(len(s))]
    for j in np.where(m)[0]:
        l = cur[j-1] if j >= 1 and cur[j-1] != MASK else None
        rr = cur[j+1] if j+1 < len(s) and cur[j+1] != MASK else None
        l2 = cur[j-2] if (l is not None and j >= 2 and cur[j-2] != MASK) else None
        pos.append((i, int(j))); ys.append(int(s[j])); lref.append(l); rref.append(rr); l2ref.append(l2)
ys = np.array(ys); N = len(ys)
keep = ys != UNK
print(f"casos: {N} (sem UNK: {keep.sum()}) | {time.time()-t0:.0f}s", flush=True)
idx_pos = {p: k for k, p in enumerate(pos)}

# ---------- NM (mix, 2M frases)
cont_flat, cont_off = carrega(f"{DIR}/treino.txt", NM_FRASES)
uni_nm = np.bincount(cont_flat, minlength=V).astype(np.float64) + 1; uni_nm /= uni_nm.sum()
boundary = cont_off[1:-1] - 1
valid = np.ones(len(cont_flat) - 1, bool); valid[boundary] = False
BI = np.bincount(cont_flat[:-1][valid] * V + cont_flat[1:][valid], minlength=V * V).reshape(V, V).astype(np.float64)
sid = np.repeat(np.arange(len(cont_off) - 1), np.diff(cont_off))
ok = (sid[:-2] == sid[1:-1]) & (sid[1:-1] == sid[2:])
keys = (cont_flat[:-2][ok] * V + cont_flat[2:][ok]) * V + cont_flat[1:-1][ok]
del sid, ok
uniq, cnt = np.unique(keys, return_counts=True); cnt = cnt.astype(np.float64)
del keys, valid, boundary

def dist_mix(l, rr):
    if l is not None and rr is not None:
        base = (l * V + rr) * V
        a = np.searchsorted(uniq, base); b = np.searchsorted(uniq, base + V)
        if b > a:
            q = np.zeros(V); q[uniq[a:b] - base] = cnt[a:b]
            return 0.8 * q / q.sum() + 0.2 * uni_nm
    if l is not None:
        row = BI[l]; s = row.sum()
        if s > 0: return 0.8 * row / s + 0.2 * uni_nm
    return uni_nm

LN = np.zeros((N, V), np.float32)
for k in range(N):
    LN[k] = np.log(dist_mix(lref[k], rref[k]) + 1e-9)
del BI, uniq, cnt
print(f"NM pronto | {time.time()-t0:.0f}s", flush=True)

# ---------- SM: 2 à esquerda -> próxima, treinada no corpus grande
sm_flat, sm_off = carrega(f"{DIR}/treino.txt", SM_FRASES)
sid = np.repeat(np.arange(len(sm_off) - 1), np.diff(sm_off))
ok = (sid[:-2] == sid[1:-1]) & (sid[1:-1] == sid[2:])
SA = sm_flat[:-2][ok]; SB = sm_flat[1:-1][ok]; SY = sm_flat[2:][ok]
del sid, ok, sm_flat
NPAR = len(SY)
print(f"SM: {NPAR} pares de treino | {time.time()-t0:.0f}s", flush=True)

Es = D_RNG.normal(0, 0.1, (V, D)).astype(np.float32)
SMW = {"W1": D_RNG.normal(0, 1/np.sqrt(2*D), (2*D, HD)).astype(np.float32), "b1": np.zeros(HD, np.float32),
       "W2": D_RNG.normal(0, 1/np.sqrt(HD), (HD, V)).astype(np.float32), "b2": np.zeros(V, np.float32),
       "Es": Es}
SA_m = {k: np.zeros_like(v) for k, v in SMW.items()}
SV_m = {k: np.zeros_like(v) for k, v in SMW.items()}
LRS = 3e-3; BS = 512
for t in range(1, SM_PASSOS + 1):
    idx = D_RNG.integers(0, NPAR, BS)
    a, b, y = SA[idx], SB[idx], SY[idx]
    h0 = np.concatenate([SMW["Es"][a], SMW["Es"][b]], 1)
    h1 = np.tanh(h0 @ SMW["W1"] + SMW["b1"])
    zz = h1 @ SMW["W2"] + SMW["b2"]; zz -= zz.max(1, keepdims=True)
    p = np.exp(zz); p /= p.sum(1, keepdims=True)
    dz = p.copy(); dz[np.arange(BS), y] -= 1; dz /= BS
    dpre = (dz @ SMW["W2"].T) * (1 - h1 * h1)
    gE = np.zeros_like(SMW["Es"])
    dh0 = dpre @ SMW["W1"].T
    np.add.at(gE, a, dh0[:, :D]); np.add.at(gE, b, dh0[:, D:])
    g = {"W2": h1.T @ dz, "b2": dz.sum(0), "W1": h0.T @ dpre, "b1": dpre.sum(0), "Es": gE}
    for k in g:
        SA_m[k] = 0.9 * SA_m[k] + 0.1 * g[k]; SV_m[k] = 0.999 * SV_m[k] + 0.001 * g[k] ** 2
        mh = SA_m[k] / (1 - 0.9 ** t); vh = SV_m[k] / (1 - 0.999 ** t)
        SMW[k] = SMW[k] - LRS * mh / (np.sqrt(vh) + 1e-8)
    if t % 500 == 0:
        perda = -np.log(p[np.arange(BS), y] + 1e-12).mean()
        print(f"  SM passo {t}: perda {perda:.3f} | {time.time()-t0:.0f}s", flush=True)

# SM nos casos com as duas palavras à esquerda conhecidas; senão fica uniforme (sem efeito no ranking)
LS = np.full((N, V), np.log(1.0 / V), np.float32)
com = [k for k in range(N) if l2ref[k] is not None]
if com:
    ca = np.array([l2ref[k] for k in com]); cb = np.array([lref[k] for k in com])
    h0 = np.concatenate([SMW["Es"][ca], SMW["Es"][cb]], 1)
    h1 = np.tanh(h0 @ SMW["W1"] + SMW["b1"])
    zz = h1 @ SMW["W2"] + SMW["b2"]; zz -= zz.max(1, keepdims=True)
    lp = zz - np.log(np.exp(zz).sum(1, keepdims=True))
    LS[com] = lp.astype(np.float32)
print(f"SM nos casos: {len(com)}/{N} com contexto | {time.time()-t0:.0f}s", flush=True)

# ---------- Nx: última predição do checkpoint, com o contexto inteiro da frase mascarada
LX = np.zeros((N, V), np.float32)
B = 64
for b0 in range(0, n_te, B):
    b1 = min(b0 + B, n_te); nb = b1 - b0
    Tok = np.full((nb, L), PAD, np.int64)
    for r_, i in enumerate(range(b0, b1)):
        s, m = sents[i]; n = len(s)
        Tok[r_, :n] = s
        Tok[r_, :n][m] = MASK
    mask = Tok == MASK
    P = np.zeros((nb, L, V), np.float32); P[mask] = uni
    for _ in range(NX_PASSES):
        F = monta(Tok, mask, P); p = passe(F); P[mask] = p
    rows = np.argwhere(mask)
    for k_row, (rr_, cc_) in enumerate(rows):
        LX[idx_pos[(b0 + rr_, int(cc_))]] = np.log(p[k_row] + 1e-9)
print(f"Nx pronto ({NX_PASSES} passes) | {time.time()-t0:.0f}s", flush=True)

# ---------- combinação log-linear e busca de pesos (metade par ajusta, metade ímpar mede)
ikeep = np.where(keep)[0]
tune, test = ikeep[::2], ikeep[1::2]

def acc(w, idx):
    a, b, c = w
    S = a * LN[idx] + b * LX[idx] + c * LS[idx]
    S[:, :3] = -1e9
    return float((S.argmax(1) == ys[idx]).mean())

grade = [(a, b, c) for a in [0, 0.5, 1] for b in [0, 0.5, 1, 2] for c in [0, 0.5, 1] if (a, b, c) != (0, 0, 0)]
melhor = max(grade, key=lambda w: acc(w, tune))
resultado = {
    "NM sozinho": acc((1, 0, 0), test),
    "Nx sozinho": acc((0, 1, 0), test),
    "SM sozinho": acc((0, 0, 1), test),
    "SM sozinho (só casos com contexto)": acc((0, 0, 1), np.array([k for k in test if l2ref[k] is not None])),
    "NM sozinho (só casos com contexto)": acc((1, 0, 0), np.array([k for k in test if l2ref[k] is not None])),
    "pesos escolhidos (a,b,c)": list(melhor),
    "junta (metade de teste)": acc(melhor, test),
    "junta (metade de ajuste, referência)": acc(melhor, tune),
    "sem NM": acc((0, melhor[1], melhor[2]), test),
    "sem Nx": acc((melhor[0], 0, melhor[2]), test),
    "sem SM": acc((melhor[0], melhor[1], 0), test),
}
print("RESUMO", json.dumps(resultado, ensure_ascii=False), flush=True)
for k, v in resultado.items():
    print(f"  {k}: {v if isinstance(v, list) else f'{v:.1%}'}", flush=True)
print(f"FIM | {time.time()-t0:.0f}s", flush=True)
