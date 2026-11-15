"""
Dupla NM + Nx no corpus grande (teste = artigos fora do treino).
- NM: contagem bidirecional (bigrama dense + skip-gram (esq,dir)->meio via np.unique/searchsorted)
- Nx: pesos salvos pelo nx_grande.py (lidos de PESOS_DIR)
- Mesma máscara do avalia() do nx_grande (RNG 7, 30%, frases com >=3 palavras)
- Ajusta pesos numa metade das posições e mede na outra
"""
import array, itertools, json, os, re, sys, time
from collections import Counter
import numpy as np

DIR = os.environ.get("DADOS", "dados_grande")
PESOS = os.environ.get("PESOS_DIR", "pesos")
NTR = int(os.environ.get("NM_FRASES", "2000000"))
NTE = int(os.environ.get("NTE", "1500"))
L = 32; PAD, UNK, MASK = 0, 1, 2; VOC = 8000
TOK = re.compile(r"\w+|[^\w\s]")
tokens = lambda s: TOK.findall(s.lower())[:L]
t0 = time.time()

cont = Counter()
with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i >= 1_000_000: break
        cont.update(tokens(l))
itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(VOC)]
stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)

def carrega(caminho, limite=None):
    flat = array.array("i"); off = array.array("q", [0])
    with open(caminho, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if limite is not None and i >= limite: break
            ids = [stoi.get(w, UNK) for w in tokens(l)]
            if len(ids) < 3: continue
            flat.extend(ids); off.append(len(flat))
    return np.frombuffer(flat, dtype=np.int32).astype(np.int64), np.frombuffer(off, dtype=np.int64).copy()

flat, off = carrega(f"{DIR}/treino.txt", NTR)
print(f"NM: {len(off)-1} frases, {len(flat)} tokens | {time.time()-t0:.0f}s", flush=True)
uni = np.bincount(flat, minlength=V).astype(np.float32) + 1; uni /= uni.sum()
# bigrama denso (pares dentro da frase)
boundary = off[1:-1] - 1
valid = np.ones(len(flat) - 1, bool); valid[boundary] = False
BI = np.bincount(flat[:-1][valid] * V + flat[1:][valid], minlength=V * V).reshape(V, V).astype(np.float32)
# skip-gram: (esq, dir) -> meio, dentro da mesma frase
sid = np.repeat(np.arange(len(off) - 1), np.diff(off))
ok = (sid[:-2] == sid[1:-1]) & (sid[1:-1] == sid[2:])
esq, meio, dire = flat[:-2][ok], flat[1:-1][ok], flat[2:][ok]
keys = (esq * V + dire) * V + meio
del esq, meio, dire, sid, ok
uniq, cnt = np.unique(keys, return_counts=True)
del keys
print(f"NM pronto: skip-gram com {len(uniq)} chaves | {time.time()-t0:.0f}s", flush=True)

def nm(l, r):
    if l is not None and r is not None:
        base = (l * V + r) * V
        a = np.searchsorted(uniq, base); b = np.searchsorted(uniq, base + V)
        if b > a:
            q = np.zeros(V, np.float32); q[uniq[a:b] - base] = cnt[a:b]
            return 0.8 * q / q.sum() + 0.2 * uni
    if l is not None:
        row = BI[l]; s = row.sum()
        if s > 0: return 0.8 * row / s + 0.2 * uni
    return uni

# ---------- Nx (pesos salvos)
pz = np.load(os.path.join(PESOS, "nx_grande.npz"))
E = pz["E"]; W = {k: pz[k] for k in ["W1", "b1", "W2", "b2", "ws", "bs"]}
D = E.shape[1]
def monta(Tok, mask, P):
    Lc = Tok.shape[1]
    X = E[Tok].copy(); X[mask] = P[mask] @ E
    Xp = np.zeros((Tok.shape[0], Lc + 4, D), np.float32); Xp[:, 2:Lc+2] = X
    F = np.concatenate([Xp[:, 0:Lc], Xp[:, 1:Lc+1], X, Xp[:, 3:Lc+3], Xp[:, 4:Lc+4]], axis=2)
    return F[mask]
def passe(F):
    H = np.tanh(F @ W["W1"] + W["b1"]); z = H @ W["W2"] + W["b2"]
    z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p

# ---------- teste: mesmas máscaras do avalia() do nx_grande
tf, to = carrega(f"{DIR}/teste.txt", NTE * 3)
n_te = min(NTE, len(to) - 1)
r = np.random.default_rng(7)
linhas = []
for b in range(0, n_te, 64):
    ids = list(range(b, min(b + 64, n_te)))
    Tok = np.full((len(ids), L), PAD, np.int64); Y = Tok.copy()
    for k, i in enumerate(ids):
        s = tf[to[i]:to[i+1]][:L]; n = len(s)
        Tok[k, :n] = s; Y[k, :n] = s
        m = r.random(n) < 0.3
        if not m.any(): m[0] = True
        Tok[k, :n][m] = MASK
    linhas.append((Tok, Y))

def lg(v):
    v = np.log(v + 1e-9).astype(np.float32); v[:3] = -1e9; return v

rows = []   # (nm, nx1, nx4, nx8, y)
for Tok, Y in linhas:
    mask = Tok == MASK
    P = np.zeros(Tok.shape + (V,), np.float32); P[mask] = uni
    saidas = {}
    for k in range(1, 9):
        p = passe(monta(Tok, mask, P)); P[mask] = p
        if k in (1, 4, 8): saidas[k] = p.copy()
    idx = np.argwhere(mask)
    for row, (b, i) in enumerate(idx):
        y = Y[b, i]
        if y == UNK: continue
        l = Tok[b, i-1] if i >= 1 and Tok[b, i-1] != MASK else None
        rr = Tok[b, i+1] if i+1 < Tok.shape[1] and Tok[b, i+1] != MASK else None
        if Y[b, i+1 if i+1 < Tok.shape[1] else i] == PAD and i+1 >= Tok.shape[1]: rr = None
        rows.append((lg(nm(l, rr)), lg(saidas[1][row]), lg(saidas[4][row]), lg(saidas[8][row]), int(y)))
print(f"posições avaliadas: {len(rows)} | {time.time()-t0:.0f}s", flush=True)
ys = np.array([x[4] for x in rows])
idx = np.arange(len(rows)); np.random.default_rng(1).shuffle(idx)
ajuste, teste = idx[:len(idx)//2], idx[len(idx)//2:]
def acc(ws, ids):
    ok = 0
    for i in ids:
        sc = sum(w * rows[i][k] for k, w in ws.items() if w)
        ok += int(np.argmax(sc) == ys[i])
    return ok / len(ids)
for k, nome in enumerate(["NM", "Nx 1 passe", "Nx 4 passes", "Nx 8 passes"]):
    print(f"{nome} sozinho (metade de teste): {acc({k: 1.0}, teste):.1%}", flush=True)
best = None
for wn, wx in itertools.product([0, 0.25, 0.5, 1.0, 2.0], [0.5, 1.0]):
    c = {0: wn, 2: wx}
    if wn == 0 and wx == 0: continue
    a = acc(c, ajuste)
    if best is None or a > best[1]: best = (c, a)
print(f"melhor dupla no ajuste: NM={best[0][0]}, Nx4={best[0][2]} -> teste {acc(best[0], teste):.1%}", flush=True)
print(f"NM + Nx4 (pesos iguais, teste): {acc({0: 1.0, 2: 1.0}, teste):.1%}", flush=True)
print("FIM", flush=True)
