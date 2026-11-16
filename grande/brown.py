"""
Modelo de classes de palavras interpolado com o NM (ideia do Brown et al. 1992).

Aproximação: as classes não vêm do algoritmo de Brown (aglomeração gulosa), e sim de k-means
sobre vetores PPMI reduzidos por SVD (método comum como substituto). Por isso o nome é
"brown-like" nos resultados.

P(w | l, r) = λ · P_palavra(w | l, r) + (1-λ) · P_classe(c(w) | l, r) · P(w | c(w))

- P_palavra: NM 'mix' (mesmo estimador do nm_v2)
- P_classe: mesmo estimador, aplicado sobre classes em vez de palavras
- Escolhe λ na metade par dos casos, mede na metade ímpar. Mesmo teste por artigo do NM.

Variáveis de ambiente: DADOS, NM_FRASES (2000000), NCLS (256), DIM (50), NTE (1500).
"""
import array, json, os, re, time
from collections import Counter
import numpy as np

DIR = os.environ.get("DADOS", "dados_grande")
NM_FRASES = int(os.environ.get("NM_FRASES", "2000000"))
NCLS = int(os.environ.get("NCLS", "256"))
DIM = int(os.environ.get("DIM", "50"))
NTE = int(os.environ.get("NTE", "1500"))
L = 32; VOC = 8000; PAD, UNK, MASK = 0, 1, 2
TOK = re.compile(r"\w+|[^\w\s]")
tokens = lambda s: TOK.findall(s.lower())[:L]
rng = np.random.default_rng(0)
t0 = time.time()

# ---------- vocabulário (igual ao NM)
cont = Counter()
with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i >= 1_000_000: break
        cont.update(tokens(l))
itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(VOC)]
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)

def carrega(caminho, limite=None):
    flat = array.array("i"); off = array.array("q", [0])
    with open(caminho, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if limite is not None and i >= limite: break
            ids = [stoi.get(w, UNK) for w in tokens(l)]
            if len(ids) < 3: continue
            flat.extend(ids); off.append(len(flat))
    return np.frombuffer(flat, dtype=np.int32).astype(np.int64), np.frombuffer(off, dtype=np.int64).copy()

flat, off = carrega(f"{DIR}/treino.txt", NM_FRASES)
print(f"treino: {len(off)-1} frases | {time.time()-t0:.0f}s", flush=True)

def contagens(seq, base):
    """unigrama, bigrama denso e skip-gram esparso (esq, dir) -> meio, sobre ids de tamanho base."""
    uni = np.bincount(seq, minlength=base).astype(np.float64) + 1; uni /= uni.sum()
    boundary = off[1:-1] - 1
    valid = np.ones(len(seq) - 1, bool); valid[boundary] = False
    BI = np.bincount(seq[:-1][valid] * base + seq[1:][valid], minlength=base * base).reshape(base, base).astype(np.float64)
    sid = np.repeat(np.arange(len(off) - 1), np.diff(off))
    ok = (sid[:-2] == sid[1:-1]) & (sid[1:-1] == sid[2:])
    keys = (seq[:-2][ok] * base + seq[2:][ok]) * base + seq[1:-1][ok]
    uniq, cnt = np.unique(keys, return_counts=True)
    return uni, BI, uniq, cnt.astype(np.float64)

def dist(modelo, base, l, rr):
    uni, BI, uniq, cnt = modelo
    if l is not None and rr is not None:
        b = (l * base + rr) * base
        a = np.searchsorted(uniq, b); c = np.searchsorted(uniq, b + base)
        if c > a:
            q = np.zeros(base); q[uniq[a:c] - b] = cnt[a:c]
            return 0.8 * q / q.sum() + 0.2 * uni
    if l is not None:
        row = BI[l]; s = row.sum()
        if s > 0: return 0.8 * row / s + 0.2 * uni
    return uni

# ---------- classes: PPMI + SVD + k-means
t_cls = time.time()
uni_w, BI_w, _, _ = contagens(flat, V)
N = BI_w.sum()
rs = BI_w.sum(1); cs = BI_w.sum(0)
ppmi = np.maximum(np.log((BI_w * N + 1e-12) / (rs[:, None] * cs[None, :] + 1e-12) + 1e-12), 0).astype(np.float32)
ppmi[BI_w == 0] = 0
del BI_w
# SVD truncada por iteração de potência (rápido, sem numpy.linalg.svd completo)
Q = rng.standard_normal((V, DIM + 10)).astype(np.float32)
for _ in range(4):
    Q, _ = np.linalg.qr(ppmi @ (ppmi.T @ Q))
B_ = Q.T @ ppmi
Ub, S, _ = np.linalg.svd(B_, full_matrices=False)
emb = (Q @ Ub[:, :DIM]) * np.sqrt(S[:DIM])
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
del ppmi, Q
# k-means
cent = emb[rng.choice(V, NCLS, replace=False)].copy()
for it in range(15):
    dist2 = emb @ cent.T
    asg = dist2.argmax(1)
    for c in range(NCLS):
        m = asg == c
        if m.any():
            cent[c] = emb[m].mean(0); cent[c] /= np.linalg.norm(cent[c]) + 1e-9
cls = asg.astype(np.int64)
cls[:3] = NCLS  # especiais em classe própria (nunca previstas)
C = NCLS + 1
tam = np.bincount(cls, minlength=C)
print(f"classes: {NCLS} | tamanhos médio {np.mean(tam[:NCLS]):.1f}, máx {tam[:NCLS].max()} | {time.time()-t_cls:.0f}s", flush=True)

# P(w | c) a partir da frequência de palavras
uni_cls_sum = np.bincount(cls, weights=uni_w, minlength=C)
P_wc = uni_w / uni_cls_sum[cls]

# ---------- modelos: palavra (NM) e classe
modelo_w = contagens(flat, V)
flat_cls = cls[flat]
modelo_c = contagens(flat_cls, C)
print(f"modelos prontos | {time.time()-t0:.0f}s", flush=True)

# ---------- teste por artigo (igual ao nm_v2: RNG 7, uma chamada por frase)
tf, to = carrega(f"{DIR}/teste.txt", NTE * 3)
n_te = min(NTE, len(to) - 1)
r = np.random.default_rng(7)
casos = []
for i in range(n_te):
    s = tf[to[i]:to[i+1]][:L]
    m = r.random(len(s)) < 0.3
    if not m.any(): m[0] = True
    cur = [MASK if m[j] else int(s[j]) for j in range(len(s))]
    for j in np.where(m)[0]:
        y = int(s[j])
        if y == UNK: continue
        l = cur[j-1] if j >= 1 and cur[j-1] != MASK else None
        rr = cur[j+1] if j+1 < len(s) and cur[j+1] != MASK else None
        casos.append((l, rr, y))
N_c = len(casos)
print(f"casos de teste: {N_c}", flush=True)

PW = np.zeros((N_c, V), np.float32)
PC = np.zeros((N_c, V), np.float32)
ys = np.array([c[2] for c in casos])
for k, (l, rr, y) in enumerate(casos):
    pw = dist(modelo_w, V, l, rr)
    cl = cls[l] if l is not None else None
    cr = cls[rr] if rr is not None else None
    pc = dist(modelo_c, C, cl, cr)
    PW[k] = pw
    PC[k] = pc[cls] * P_wc  # P(classe(w)|l,r) * P(w|classe(w))
PW[:, :3] = 0; PC[:, :3] = 0
print(f"probabilidades prontas | {time.time()-t0:.0f}s", flush=True)

def acc(lam, idx):
    S = lam * PW[idx] + (1 - lam) * PC[idx]
    return float((S.argmax(1) == ys[idx]).mean())

ikeep = np.arange(N_c)
tune, test = ikeep[::2], ikeep[1::2]
lams = [round(x, 2) for x in np.linspace(0, 1, 11)]
melhor = max(lams, key=lambda x: acc(x, tune))
resultado = {
    "NCLS": NCLS, "DIM": DIM, "casos": N_c,
    "NM sozinho (metade de teste)": acc(1.0, test),
    "classes sozinhas (metade de teste)": acc(0.0, test),
    "lambda escolhido": melhor,
    "interpolado (metade de teste)": acc(melhor, test),
    "ganho sobre NM": acc(melhor, test) - acc(1.0, test),
}
print("RESUMO", json.dumps(resultado, ensure_ascii=False), flush=True)
for k, v in resultado.items():
    if isinstance(v, float) and k != "lambda escolhido":
        v = f"{v:+.2%}" if k == "ganho sobre NM" else f"{v:.2%}"
    print(f"  {k}: {v}", flush=True)
print(f"FIM | {time.time()-t0:.0f}s", flush=True)
