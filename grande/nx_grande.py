"""
Nx em escala: dados lidos de arquivo (um por linha), guardados como arrays numpy
(não listas Python, pra caber na RAM do runner). Mesmo modelo do nx.py:
palpite macio refinado por passes, janela 2+2, geração com compromisso.
Variáveis de ambiente: DADOS, PASSOS, SALVA_A_CADA, PESOS_DIR.
"""
import array, json, os, re, sys, time
from collections import Counter
import numpy as np

DIR = os.environ.get("DADOS", "dados_grande")
PASSOS = int(os.environ.get("PASSOS", "100000"))
SALVA = int(os.environ.get("SALVA_A_CADA", "20000"))
PESOS = os.environ.get("PESOS_DIR", "pesos")
L = 32; D = 64; HD = 128; B = 16; LR = 2e-3; K = 8; VOC = 8000
PAD, UNK, MASK = 0, 1, 2
TOK = re.compile(r"\w+|[^\w\s]")
rng = np.random.default_rng(0)
t0 = time.time()

def tokens(s):
    return TOK.findall(s.lower())[:L]

# ---------- vocabulário: amostra de 1 milhão de frases de treino
cont = Counter()
with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i >= 1_000_000:
            break
        cont.update(tokens(l))
itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(VOC)]
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)
print(f"vocabulário: {V} | {time.time()-t0:.0f}s", flush=True)

def carrega(caminho):
    flat = array.array("i"); off = array.array("q", [0])
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            ids = [stoi.get(w, UNK) for w in tokens(l)]
            if len(ids) < 3:
                continue
            flat.extend(ids); off.append(len(flat))
    return np.frombuffer(flat, dtype=np.int32).copy(), np.frombuffer(off, dtype=np.int64).copy()

flat_tr, off_tr = carrega(f"{DIR}/treino.txt")
flat_te, off_te = carrega(f"{DIR}/teste.txt")
uni = np.bincount(flat_tr, minlength=V).astype(np.float32) + 1
uni /= uni.sum()
print(f"treino: {len(off_tr)-1} frases, {len(flat_tr)} tokens | teste: {len(off_te)-1} frases | {time.time()-t0:.0f}s", flush=True)

E = rng.normal(0, 0.1, (V, D)).astype(np.float32); E[PAD] = 0
W = {"W1": rng.normal(0, 1/np.sqrt(5*D), (5*D, HD)).astype(np.float32), "b1": np.zeros(HD, np.float32),
     "W2": rng.normal(0, 1/np.sqrt(HD), (HD, V)).astype(np.float32), "b2": np.zeros(V, np.float32),
     "ws": np.zeros(HD, np.float32), "bs": np.zeros(1, np.float32)}
A = {k: [np.zeros_like(v), np.zeros_like(v), 0] for k, v in W.items()}

def monta(Tok, mask, P):
    Lc = Tok.shape[1]
    Pm = P[mask]
    X = E[Tok].copy()
    X[mask] = Pm @ E
    Bn = Tok.shape[0]
    Xp = np.zeros((Bn, Lc + 4, D), np.float32); Xp[:, 2:Lc+2] = X
    F = np.concatenate([Xp[:, 0:Lc], Xp[:, 1:Lc+1], X, Xp[:, 3:Lc+3], Xp[:, 4:Lc+4]], axis=2)
    return F[mask]

def passe(F):
    H = np.tanh(F @ W["W1"] + W["b1"])
    z = H @ W["W2"] + W["b2"]
    z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    s = 1 / (1 + np.exp(-(H @ W["ws"] + W["bs"][0])))
    return H, p, s

def lote_treino():
    idx = rng.integers(0, len(off_tr) - 1, B)
    Tok = np.full((B, L), PAD, np.int64); Y = Tok.copy()
    for r_, i in enumerate(idx):
        s = flat_tr[off_tr[i]:off_tr[i+1]][:L]
        n = len(s)
        Tok[r_, :n] = s; Y[r_, :n] = s
        m = rng.random(n) < rng.uniform(0.3, 0.6)
        if not m.any():
            m[rng.integers(n)] = True
        Tok[r_, :n][m] = MASK
    return Tok, Tok == MASK, Y

def adam(g):
    for k in g:
        m, v, t = A[k]; t += 1
        m[:] = 0.9*m + 0.1*g[k]; v[:] = 0.999*v + 0.001*g[k]**2
        W[k] -= LR * (m/(1-0.9**t)) / (np.sqrt(v/(1-0.999**t)) + 1e-8)
        A[k][2] = t

def treina():
    perda = 0.0; n = 0; t1 = time.time()
    for passo in range(1, PASSOS + 1):
        Tok, mask, Y = lote_treino()
        P = np.zeros((B, L, V), np.float32); P[mask] = uni
        t = int(rng.integers(0, K))
        for _ in range(t):
            F = monta(Tok, mask, P); _, p, _ = passe(F); P[mask] = p
        F = monta(Tok, mask, P)
        H, p, s = passe(F)
        y = Y[mask]; M = len(y)
        perda += -np.log(p[np.arange(M), y] + 1e-12).mean(); n += 1
        acerto = (p.argmax(1) == y).astype(np.float32)
        dz = p.copy(); dz[np.arange(M), y] -= 1; dz /= M
        ds = (s - acerto) / M
        dH = dz @ W["W2"].T + np.outer(ds, W["ws"])
        dpre = dH * (1 - H * H)
        adam({"W2": H.T @ dz, "b2": dz.sum(0), "W1": F.T @ dpre, "b1": dpre.sum(0),
              "ws": H.T @ ds, "bs": np.array([ds.sum()], np.float32)})
        if passo % 500 == 0:
            print(f"  passo {passo}/{PASSOS}: perda {perda/n:.3f} | {time.time()-t1:.0f}s", flush=True)
            perda = 0.0; n = 0
        if passo % SALVA == 0 or passo == PASSOS:
            salva(passo)

def salva(passo):
    os.makedirs(PESOS, exist_ok=True)
    np.savez(os.path.join(PESOS, "nx_grande.npz"), E=E, uni=uni, **W)
    json.dump({"passo": passo, "itos": itos}, open(os.path.join(PESOS, "nx_grande.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  pesos salvos no passo {passo}", flush=True)

def avalia(passos_fixos=None, n_frases=2000):
    r = np.random.default_rng(7)
    ok = tot = 0; usados = []
    n_frases = min(n_frases, len(off_te) - 1)
    for b in range(0, n_frases, 64):
        ids = list(range(b, min(b + 64, n_frases)))
        Tok = np.full((len(ids), L), PAD, np.int64); Y = Tok.copy()
        for r_, i in enumerate(ids):
            s = flat_te[off_te[i]:off_te[i+1]][:L]; n = len(s)
            Tok[r_, :n] = s; Y[r_, :n] = s
            m = r.random(n) < 0.3
            if not m.any(): m[0] = True
            Tok[r_, :n][m] = MASK
        mask = Tok == MASK
        P = np.zeros((len(ids), L, V), np.float32); P[mask] = uni
        y = Y[mask]
        for k in range(K):
            F = monta(Tok, mask, P); _, p, _ = passe(F); P[mask] = p
            if passos_fixos is not None and k + 1 == passos_fixos:
                break
        keep = y != UNK
        pred = p[:, 3:].argmax(1) + 3
        ok += int((pred == y)[keep].sum()); tot += int(keep.sum())
    return ok / max(1, tot)

def gera_commit(n=40, Lc=16, passos=6, temp=0.9, topk=30, rep=0.6, seed=3):
    r = np.random.default_rng(seed)
    Tok = np.full((n, Lc), MASK, np.int64); mask = Tok == MASK
    P = np.zeros((n, Lc, V), np.float32); P[mask] = uni
    for k in range(passos):
        if not mask.any(): break
        pos = np.argwhere(mask)
        F = monta(Tok, mask, P); _, p, _ = passe(F)
        q = p[:, 3:].copy()
        for row, (b, _) in enumerate(pos):
            for u in Tok[b][~mask[b]]:
                if u >= 3: q[row, u - 3] *= rep
        thr = -np.sort(-q, axis=1)[:, topk-1:topk]
        q = np.where(q >= thr, q, 0)
        q = np.power(q + 1e-12, 1.0 / temp); q /= q.sum(1, keepdims=True)
        escolha = (q.cumsum(1) < r.random((len(q), 1))).sum(1) + 3
        conf = p[:, 3:].max(1)
        for b in range(n):
            rows = np.where(pos[:, 0] == b)[0]
            if len(rows) == 0: continue
            quant = max(1, int(np.ceil(len(rows) / (passos - k))))
            for row in rows[np.argsort(-conf[rows])][:quant]:
                c = pos[row, 1]
                Tok[b, c] = escolha[row]; mask[b, c] = False
        P[pos[:, 0], pos[:, 1]] = p
    return [" ".join(itos[int(w)] for w in Tok[b]) for b in range(n)]

if __name__ == "__main__":
    print(f"treinando {PASSOS} passos (lote {B} frases)", flush=True)
    treina()
    print(f"treino terminou em {time.time()-t0:.0f}s", flush=True)
    for k in [1, 4, 8]:
        print(f"  acerto mascarado ({k} passes): {avalia(passos_fixos=k):.1%}", flush=True)
    am = gera_commit()
    print(f"geração: distintas {len(set(am))}/{len(am)}", flush=True)
    for t in am[:12]:
        print("   >", t, flush=True)
    print("FIM", flush=True)
