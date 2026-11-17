"""
Híbrido pequeno (palavras inteiras, sem transformer, LSTM ou GRU de verdade):
- atenção leve sobre a janela (softmax de score por posição, score = X·q + a_k)  [ideia do transformer]
- portão por posição (sigmoid(X·u))                                               [ideia da LSTM/GRU]
- prior do NM somado aos logits, com peso β aprendido                             [contagem como entrada]
- refinamento por passes (difusão), como no Nx

Protocolo:
- NM é construído com as primeiras NM_FRASES linhas; o MLP treina com NTREINO frases seguintes.
  Assim o NM não decora o que o MLP vê no treino (evita vazamento).
- Teste: mesmo das variantes (1500 frases, RNG 7, 30% por palavra). Métrica: palavra inteira,
  UNK conta como erro, e versão sem UNK como referência.

Variáveis de ambiente: DADOS, PASSOS, JANELA, NTE, NM_FRASES, NTREINO, NM (0|1), ATENCAO (0|1).
"""
import array, json, os, re, time, resource
from collections import Counter
import numpy as np

DIR = os.environ.get("DADOS", "dados_grande")
PASSOS = int(os.environ.get("PASSOS", "1000"))
J = int(os.environ.get("JANELA", "2"))
NTE = int(os.environ.get("NTE", "1500"))
NM_FRASES = int(os.environ.get("NM_FRASES", "1500000"))
NTREINO = int(os.environ.get("NTREINO", "300000"))
USA_NM = int(os.environ.get("NM", "1"))
ATENCAO = int(os.environ.get("ATENCAO", "1"))
D = 64; HD = 128; B = 16; LR = 2e-3; K = 8; VOC = 8000; LW = 32; LP_MAX = 32
PAD, UNK, MASK = 0, 1, 2
K2 = 2 * J + 1
TOKR = re.compile(r"\w+|[^\w\s]")
rng = np.random.default_rng(0)
t0 = time.time()

def palavras(s):
    return TOKR.findall(s.lower())[:LW]

# ---------- vocabulário (1M primeiras linhas, igual aos outros testes)
cont = Counter()
with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i >= 1_000_000: break
        cont.update(palavras(l))
itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(VOC)]
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)

def ids_de(linha):
    ws = palavras(linha)
    if len(ws) < 3: return None
    return np.array([stoi.get(w, UNK) for w in ws], np.int64)

# ---------- NM: contagens com as primeiras NM_FRASES frases (separadas do treino do MLP)
flat_nm = array.array("i"); off_nm = array.array("q", [0])
with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i >= NM_FRASES: break
        s = ids_de(l)
        if s is None: continue
        flat_nm.extend(s.tolist()); off_nm.append(len(flat_nm))
flat = np.frombuffer(flat_nm, dtype=np.int32).astype(np.int64)
off = np.frombuffer(off_nm, dtype=np.int64).copy()
uni_nm = np.bincount(flat, minlength=V).astype(np.float64) + 1; uni_nm /= uni_nm.sum()
boundary = off[1:-1] - 1
valid = np.ones(len(flat) - 1, bool); valid[boundary] = False
BI = np.bincount(flat[:-1][valid] * V + flat[1:][valid], minlength=V * V).reshape(V, V).astype(np.float64)
sid = np.repeat(np.arange(len(off) - 1), np.diff(off))
ok = (sid[:-2] == sid[1:-1]) & (sid[1:-1] == sid[2:])
keys = (flat[:-2][ok] * V + flat[2:][ok]) * V + flat[1:-1][ok]
uniq, cnt = np.unique(keys, return_counts=True); cnt = cnt.astype(np.float64)
del sid, ok, keys, valid, boundary
print(f"NM pronto ({len(off)-1} frases) | {time.time()-t0:.0f}s", flush=True)

def nm_dist(l, rr):
    if l is not None and rr is not None:
        b = (l * V + rr) * V
        a = np.searchsorted(uniq, b); c = np.searchsorted(uniq, b + V)
        if c > a:
            q = np.zeros(V); q[uniq[a:c] - b] = cnt[a:c]
            return 0.8 * q / q.sum() + 0.2 * uni_nm
    if l is not None:
        row = BI[l]; s = row.sum()
        if s > 0: return 0.8 * row / s + 0.2 * uni_nm
    return uni_nm

def prior(Tok, mask):
    """log P_NM(w | vizinhos conhecidos) para cada posição mascarada, na ordem de mask."""
    rs, cs = np.nonzero(mask)
    out = np.zeros((len(rs), V), np.float32)
    Lb = Tok.shape[1]
    for k, (b, j) in enumerate(zip(rs, cs)):
        l = int(Tok[b, j-1]) if j >= 1 and Tok[b, j-1] not in (MASK, PAD) else None
        rr = int(Tok[b, j+1]) if j+1 < Lb and Tok[b, j+1] not in (MASK, PAD) else None
        out[k] = np.log(nm_dist(l, rr) + 1e-9)
    return out

# ---------- dados do MLP: frases logo depois das usadas pelo NM
TR = []
with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i < NM_FRASES: continue
        if i >= NM_FRASES + NTREINO: break
        s = ids_de(l)
        if s is not None: TR.append(s)
TE = []
with open(f"{DIR}/teste.txt", encoding="utf-8") as f:
    for l in f:
        s = ids_de(l)
        if s is None: continue
        TE.append(s)
        if len(TE) >= NTE: break
uni_tr = np.bincount(np.concatenate(TR), minlength=V).astype(np.float32) + 1; uni_tr /= uni_tr.sum()
print(f"treino MLP {len(TR)} frases | teste {len(TE)} | {time.time()-t0:.0f}s", flush=True)

# ---------- parâmetros
W = {"E": rng.normal(0, 0.1, (V, D)).astype(np.float32),
     "q": rng.normal(0, 0.1, D).astype(np.float32),
     "u": rng.normal(0, 0.1, D).astype(np.float32),
     "a": np.zeros(K2, np.float32),
     "W1": rng.normal(0, 1/np.sqrt(2*D), (2*D, HD)).astype(np.float32), "b1": np.zeros(HD, np.float32),
     "W2": rng.normal(0, 1/np.sqrt(HD), (HD, V)).astype(np.float32), "b2": np.zeros(V, np.float32)}
if USA_NM:
    W["beta"] = np.ones(1, np.float32)
W["E"][PAD] = 0
A = {k: [np.zeros_like(v), np.zeros_like(v), 0] for k, v in W.items()}

def sigm(x):
    return 1 / (1 + np.exp(-x))

def forward(Tok, mask, P, LPm):
    E = W["E"]; Lb = Tok.shape[1]
    X = E[Tok].copy()
    X[mask] = P[mask] @ E
    Xp = np.zeros((Tok.shape[0], Lb + 2 * J, D), np.float32); Xp[:, J:J+Lb] = X
    Xw = np.stack([Xp[:, k:k+Lb] for k in range(K2)], axis=2)   # (B,Lb,K2,D)
    Xwm = Xw[mask]                                               # (M,K2,D)
    if ATENCAO:
        s = Xwm @ W["q"] + W["a"][None, :]
        s -= s.max(1, keepdims=True); al = np.exp(s); al /= al.sum(1, keepdims=True)
        g = sigm(Xwm @ W["u"])
        v = al * g
        pooled = (v[:, :, None] * Xwm).sum(1)
        cache = (al, g)
    else:
        pooled = Xwm.mean(1); cache = None
    F = np.concatenate([pooled, Xwm[:, J]], axis=1)
    H = np.tanh(F @ W["W1"] + W["b1"])
    z = H @ W["W2"] + W["b2"]
    if USA_NM:
        z = z + W["beta"][0] * LPm
    z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return (Xwm, cache), F, H, p

def passe_p(Tok, mask, P, LPm):
    _, F, _, p = forward(Tok, mask, P, LPm)
    return p

def backward(Tok, mask, P, LPm, cache_fwd, dz):
    (Xwm, cache), F, H, _ = cache_fwd
    g = {}
    dH = dz @ W["W2"].T
    g["W2"] = H.T @ dz; g["b2"] = dz.sum(0)
    if USA_NM:
        g["beta"] = np.array([(dz * LPm).sum()], np.float32)
    dpre = dH * (1 - H * H)
    g["W1"] = F.T @ dpre; g["b1"] = dpre.sum(0)
    dF = dpre @ W["W1"].T
    dpooled = dF[:, :D]; dXself = dF[:, D:]
    M = Xwm.shape[0]
    if ATENCAO:
        al, gt = cache
        v_ = al * gt
        dv = (Xwm * dpooled[:, None, :]).sum(2)                 # (M,K2)
        dXw = v_[:, :, None] * dpooled[:, None, :]
        dalpha = gt * dv; dg = al * dv
        ds = al * (dalpha - (al * dalpha).sum(1, keepdims=True))
        g["q"] = (ds[:, :, None] * Xwm).sum((0, 1))
        g["a"] = ds.sum(0)
        dXw += ds[:, :, None] * W["q"][None, None, :]
        dzg = dg * gt * (1 - gt)
        g["u"] = (dzg[:, :, None] * Xwm).sum((0, 1))
        dXw += dzg[:, :, None] * W["u"][None, None, :]
    else:
        dXw = np.repeat(dpooled[:, None, :] / K2, K2, axis=1)
    dXw[:, J] += dXself
    dXw_full = np.zeros((Tok.shape[0], Tok.shape[1], K2, D), np.float32)
    dXw_full[mask] = dXw
    dXp = np.zeros((Tok.shape[0], Tok.shape[1] + 2 * J, D), np.float32)
    for k in range(K2):
        dXp[:, k:k+Tok.shape[1]] += dXw_full[:, :, k]
    dX = dXp[:, J:J+Tok.shape[1]]
    dE = np.zeros_like(W["E"])
    np.add.at(dE, Tok[~mask], dX[~mask])
    dE += P[mask].T @ dX[mask]
    dE[PAD] = 0
    g["E"] = dE
    return g

def adam(g):
    for k in g:
        m, v, t = A[k]; t += 1
        m[:] = 0.9*m + 0.1*g[k]; v[:] = 0.999*v + 0.001*g[k]**2
        W[k] -= LR * (m/(1-0.9**t)) / (np.sqrt(v/(1-0.999**t)) + 1e-8)
        A[k][2] = t

def lote_treino():
    idx = rng.integers(0, len(TR), B)
    Lb = max(len(TR[i]) for i in idx)
    Tok = np.full((B, Lb), PAD, np.int64); mask = np.zeros((B, Lb), bool)
    for r_, i in enumerate(idx):
        s = TR[i]; n = len(s)
        Tok[r_, :n] = s
        m = rng.random(n) < rng.uniform(0.3, 0.6)
        if not m.any(): m[rng.integers(n)] = True
        mask[r_, :n] = m
    Y = Tok.copy()
    Tok[mask] = MASK
    return Tok, mask, Y

def treina():
    tt = time.time(); perda = 0.0; n = 0
    for passo in range(1, PASSOS + 1):
        Tok, mask, Y = lote_treino()
        LPm = prior(Tok, mask) if USA_NM else np.zeros((mask.sum(), V), np.float32)
        P = np.zeros((B, Tok.shape[1], V), np.float32); P[mask] = uni_tr
        t = int(rng.integers(0, K))
        for _ in range(t):
            p = passe_p(Tok, mask, P, LPm); P[mask] = p
        cf = forward(Tok, mask, P, LPm)
        p = cf[3]
        y = Y[mask]; M = len(y)
        perda += -np.log(p[np.arange(M), y] + 1e-12).mean(); n += 1
        dz = p.copy(); dz[np.arange(M), y] -= 1; dz /= M
        adam(backward(Tok, mask, P, LPm, cf, dz))
        if passo % 200 == 0:
            print(f"  passo {passo}/{PASSOS}: perda {perda/n:.3f} | {time.time()-tt:.0f}s", flush=True)
            perda = 0.0; n = 0
    return time.time() - tt

def avalia_nm_sozinho():
    """controle: argmax só do prior do NM (mesmo teste, mesmas máscaras)."""
    r = np.random.default_rng(7)
    ok_t = tot = ok_s = tot_s = 0
    for b0 in range(0, len(TE), 64):
        grupo = TE[b0:b0+64]
        Lb = max(len(s) for s in grupo)
        Tok = np.full((len(grupo), Lb), PAD, np.int64); Y = Tok.copy(); mask = np.zeros_like(Tok, bool)
        for r_, s in enumerate(grupo):
            n = len(s); Tok[r_, :n] = s; Y[r_, :n] = s
            m = r.random(n) < 0.3
            if not m.any(): m[0] = True
            mask[r_, :n] = m
        Tok[mask] = MASK
        LPm = prior(Tok, mask)
        LPm[:, :3] = -1e9
        pred = LPm.argmax(1)
        gold = Y[mask]; cert = pred == gold
        ok_t += int(cert.sum()); tot += len(gold)
        sem = gold != UNK
        ok_s += int(cert[sem].sum()); tot_s += int(sem.sum())
    return ok_t / max(1, tot), ok_s / max(1, tot_s)

def avalia(passos=4):
    r = np.random.default_rng(7)
    ok_t = tot = ok_s = tot_s = 0
    for b0 in range(0, len(TE), 64):
        grupo = TE[b0:b0+64]
        Lb = max(len(s) for s in grupo)
        Tok = np.full((len(grupo), Lb), PAD, np.int64); Y = Tok.copy(); mask = np.zeros_like(Tok, bool)
        for r_, s in enumerate(grupo):
            n = len(s); Tok[r_, :n] = s; Y[r_, :n] = s
            m = r.random(n) < 0.3
            if not m.any(): m[0] = True
            mask[r_, :n] = m
        Tok[mask] = MASK
        LPm = prior(Tok, mask) if USA_NM else np.zeros((mask.sum(), V), np.float32)
        P = np.zeros((len(grupo), Lb, V), np.float32); P[mask] = uni_tr
        for _ in range(passos):
            p = passe_p(Tok, mask, P, LPm); P[mask] = p
        pred = p[:, 3:].argmax(1) + 3
        gold = Y[mask]; cert = pred == gold
        ok_t += int(cert.sum()); tot += len(gold)
        sem = gold != UNK
        ok_s += int(cert[sem].sum()); tot_s += int(sem.sum())
    return ok_t / max(1, tot), ok_s / max(1, tot_s)

if __name__ == "__main__":
    t_tr = treina()
    print(f"treino terminou em {time.time()-t0:.0f}s (treino {t_tr:.0f}s)", flush=True)
    n_params = int(sum(v.size for v in W.values()))
    a1, _ = avalia(1)
    a4, a4s = avalia(4)
    nm_t, nm_s = avalia_nm_sozinho()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    resumo = {"NM": USA_NM, "ATENCAO": ATENCAO, "janela": J, "passos": PASSOS, "params": n_params,
              "palavra_4_passes": round(a4, 4), "palavra_sem_unk_4_passes": round(a4s, 4),
              "palavra_1_passe": round(a1, 4), "NM_sozinho_palavra": round(nm_t, 4), "NM_sozinho_sem_unk": round(nm_s, 4),
              "treino_s": round(t_tr), "pico_ram_MB": round(rss)}
    print("RESUMO", json.dumps(resumo), flush=True)
    print("FIM", flush=True)
