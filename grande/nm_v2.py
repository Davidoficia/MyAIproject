"""
NM v2: testa tamanho do treino e suavização no mesmo teste por artigo (1500 frases, RNG 7, 30%).
Estimadores para P(meio | esq, dir):
  mix  : 0.8*contagem + 0.2*unigrama, com bigrama de fallback (versão atual)
  disc : desconto absoluto (d=0.75) com backoff interpolado: skip-gram -> bigrama -> unigrama
Tamanhos de treino: NM_TAMANHOS="2000000,6000000"
"""
import array, json, os, re, sys, time
from collections import Counter
import numpy as np

DIR = os.environ.get("DADOS", "dados_grande")
TAMANHOS = [int(x) for x in os.environ.get("NM_TAMANHOS", "2000000,6000000").split(",")]
NTE = int(os.environ.get("NTE", "1500"))
D_DESC = 0.75
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
print(f"casos de teste: {len(casos)} | {time.time()-t0:.0f}s", flush=True)

def treino_nm(ntr):
    flat, off = carrega(f"{DIR}/treino.txt", ntr)
    uni = np.bincount(flat, minlength=V).astype(np.float64) + 1; uni /= uni.sum()
    boundary = off[1:-1] - 1
    valid = np.ones(len(flat) - 1, bool); valid[boundary] = False
    BI = np.bincount(flat[:-1][valid] * V + flat[1:][valid], minlength=V * V).reshape(V, V).astype(np.float64)
    sid = np.repeat(np.arange(len(off) - 1), np.diff(off))
    ok = (sid[:-2] == sid[1:-1]) & (sid[1:-1] == sid[2:])
    keys = (flat[:-2][ok] * V + flat[2:][ok]) * V + flat[1:-1][ok]
    del sid, ok
    uniq, cnt = np.unique(keys, return_counts=True)
    del keys
    return uni, BI, uniq, cnt.astype(np.float64)

def dist_mix(modelo, l, rr):
    uni, BI, uniq, cnt = modelo
    if l is not None and rr is not None:
        base = (l * V + rr) * V
        a = np.searchsorted(uniq, base); b = np.searchsorted(uniq, base + V)
        if b > a:
            q = np.zeros(V); q[uniq[a:b] - base] = cnt[a:b]
            return 0.8 * q / q.sum() + 0.2 * uni
    if l is not None:
        row = BI[l]; s = row.sum()
        if s > 0: return 0.8 * row / s + 0.2 * uni
    return uni

def dist_disc(modelo, l, rr):
    uni, BI, uniq, cnt = modelo
    p_bi = uni
    if l is not None:
        row = BI[l]; s = row.sum()
        if s > 0:
            n1 = (row > 0).sum()
            p_bi = np.maximum(row - D_DESC, 0) / s + (D_DESC * n1 / s) * uni
    if l is not None and rr is not None:
        base = (l * V + rr) * V
        a = np.searchsorted(uniq, base); b = np.searchsorted(uniq, base + V)
        if b > a:
            c = cnt[a:b]; tot = c.sum(); idx = uniq[a:b] - base
            p = np.zeros(V)
            p[idx] = np.maximum(c - D_DESC, 0) / tot
            return p + (D_DESC * len(c) / tot) * p_bi
    return p_bi

resultados = {}
for ntr in TAMANHOS:
    modelo = treino_nm(ntr)
    print(f"treino NM com {ntr} frases pronto | {time.time()-t0:.0f}s", flush=True)
    for nome, fn in [("mix", dist_mix), ("disc", dist_disc)]:
        ok = 0
        for l, rr, y in casos:
            q = fn(modelo, l, rr).copy(); q[:3] = 0
            ok += int(np.argmax(q) == y)
        resultados[f"{nome}-{ntr}"] = ok / len(casos)
        print(f"  NM {nome} com {ntr} frases: {ok/len(casos):.1%}", flush=True)
    del modelo
print("RESUMO", json.dumps(resultados), flush=True)
print("FIM", flush=True)
