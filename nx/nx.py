"""
Nx: difusão com refinamento iterativo e parada adaptativa (numpy, sem autograd).
- Estado = distribuição sobre vocabulário nas posições mascaradas (P), que é refinada a cada passe.
- Cada passe vê a janela (2 à esq, 2 à dir) com os tokens conhecidos e o palpite atual (P@E) dos mascarados.
- Como cada passe enxerga só 2 vizinhos, K passes dão alcance de ~2K palavras.
- Cabeça de parada: prevê se o palpite atual já está correto. O modelo para quando isso fica alto.
- Treino: passes sem gradiente (t aleatório) e um passe final com gradiente (E congelado).
"""
import json, os, re, sys, time
import numpy as np

W_DIR = os.environ.get("NX_DADOS", "/tmp/wchk/wiki_dados") + "/"
TOK = re.compile(r"\w+|[^\w\s]")
tok = lambda s: TOK.findall(s.lower())
TEMAS = ["animais", "cidades", "historia", "ciencia"]
TESTE = sys.argv[1] if len(sys.argv) > 1 else "animais"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
K = 8; L = 32; D = 64; HD = 128; B = 16; LR = 2e-3; PMASK = (0.3, 0.6)
rng = np.random.default_rng(0)

lidos = {t: [l.strip() for l in open(f"{W_DIR}wiki_{t}.txt", encoding="utf-8") if l.strip()] for t in TEMAS}
train = [tok(f)[:L] for t in TEMAS if t != TESTE for f in lidos[t]]
test = [tok(f)[:L] for f in lidos[TESTE][:640]]
from collections import Counter
cont = Counter(w for s in train for w in s)
itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(8000)]
stoi = {w: i for i, w in enumerate(itos)}
PAD, UNK, MASK = 0, 1, 2; V = len(itos)
enc = lambda s: [stoi.get(w, UNK) for w in s]
train_ids = [enc(s) for s in train]; test_ids = [enc(s) for s in test]
uni = np.bincount([w for s in train_ids for w in s], minlength=V).astype(np.float32) + 1
uni /= uni.sum()

E = rng.normal(0, 0.1, (V, D)).astype(np.float32); E[PAD] = 0
W = {"W1": rng.normal(0, 1/np.sqrt(5*D), (5*D, HD)).astype(np.float32), "b1": np.zeros(HD, np.float32),
     "W2": rng.normal(0, 1/np.sqrt(HD), (HD, V)).astype(np.float32), "b2": np.zeros(V, np.float32),
     "ws": np.zeros(HD, np.float32), "bs": np.zeros(1, np.float32)}
A = {k: [np.zeros_like(v), np.zeros_like(v), 0] for k, v in W.items()}

def monta(Tok, mask, P):
    """Tok (B,L) ids; mask (B,L) bool; P (B,L,V) palpites. Devolve features dos mascarados."""
    Pm = P[mask]                          # (M,V)
    X = E[Tok].copy()
    X[mask] = Pm @ E                      # palpite atual vira embedding macio
    Bn = Tok.shape[0]
    Xp = np.zeros((Bn, L + 4, D), np.float32); Xp[:, 2:L+2] = X
    F = np.concatenate([Xp[:, 0:L], Xp[:, 1:L+1], X, Xp[:, 3:L+3], Xp[:, 4:L+4]], axis=2)
    return F[mask]

def passe(F):
    H = np.tanh(F @ W["W1"] + W["b1"])
    z = H @ W["W2"] + W["b2"]
    z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    s = 1 / (1 + np.exp(-(H @ W["ws"] + W["bs"][0])))
    return H, p, s

def corrompe(lotes):
    Tok = np.full((len(lotes), L), PAD, np.int64); Y = np.full((len(lotes), L), PAD, np.int64)
    for i, s in enumerate(lotes):
        Tok[i, :len(s)] = s; Y[i, :len(s)] = s
        r = rng.uniform(*PMASK)
        m = rng.random(len(s)) < r
        if not m.any(): m[rng.integers(len(s))] = True
        Tok[i, :len(s)][m] = MASK
    mask = (Tok == MASK)
    return Tok, mask, Y

def adam(g):
    for k in g:
        m, v, t = A[k]; t += 1
        m[:] = 0.9*m + 0.1*g[k]; v[:] = 0.999*v + 0.001*g[k]**2
        mh = m/(1-0.9**t); vh = v/(1-0.999**t)
        W[k] -= LR * mh / (np.sqrt(vh) + 1e-8); A[k][2] = t

def treina():
    t0 = time.time(); perda_media = 0; n = 0
    for step in range(STEPS):
        idx = rng.integers(0, len(train_ids), size=B)
        Tok, mask, Y = corrompe([train_ids[i] for i in idx])
        P = np.zeros((B, L, V), np.float32); P[mask] = uni
        t = int(rng.integers(0, K))
        for _ in range(t):
            F = monta(Tok, mask, P); _, p, _ = passe(F); P[mask] = p
        F = monta(Tok, mask, P)
        H, p, s = passe(F)
        y = Y[mask]; M = len(y)
        loss = -np.log(p[np.arange(M), y] + 1e-12).mean()
        acerto = (p.argmax(1) == y).astype(np.float32)
        dz = p.copy(); dz[np.arange(M), y] -= 1; dz /= M
        ds = ((s - acerto) / M)                      # BCE da cabeça de parada
        dH = dz @ W["W2"].T + np.outer(ds, W["ws"])
        dpre = dH * (1 - H * H)
        g = {"W2": H.T @ dz, "b2": dz.sum(0), "W1": F.T @ dpre, "b1": dpre.sum(0),
             "ws": H.T @ ds, "bs": np.array([ds.sum()], np.float32)}
        adam(g)
        perda_media += loss; n += 1
        if (step + 1) % 250 == 0:
            print(f"  passo {step+1}/{STEPS}: perda {perda_media/n:.3f} | {time.time()-t0:.0f}s", flush=True)
            perda_media = 0; n = 0

def avalia(passos_fixos=None, limiar=None):
    r = np.random.default_rng(7)
    ok = tot = 0; passes_usados = []
    for b in range(0, len(test_ids), 64):
        lotes = test_ids[b:b+64]
        Tok = np.full((len(lotes), L), PAD, np.int64); Y = Tok.copy()
        for i, s in enumerate(lotes):
            Tok[i, :len(s)] = s; Y[i, :len(s)] = s
            m = r.random(len(s)) < 0.3
            if not m.any(): m[0] = True
            Tok[i, :len(s)][m] = MASK
        mask = Tok == MASK
        P = np.zeros((len(lotes), L, V), np.float32); P[mask] = uni
        y = Y[mask]
        anterior = None
        for k in range(K):
            F = monta(Tok, mask, P); _, p, s = passe(F); P[mask] = p
            atual = p[:, 3:].argmax(1)
            if passos_fixos is not None and k + 1 == passos_fixos: break
            if limiar is not None and anterior is not None:
                if (atual != anterior).mean() <= limiar: break
            anterior = atual
        cont_ = y != UNK                       # alvo fora do vocabulário não conta (senão UNK ganha crédito)
        ok += int(((p[:, 3:].argmax(1) + 3) == y)[cont_].sum()); tot += int(cont_.sum()); passes_usados.append(k + 1)
    return ok / tot, float(np.mean(passes_usados))

def gera(n=12, L_=20, limiar=0.8, seed=0):
    r = np.random.default_rng(seed)
    Tok = np.full((n, L_), MASK, np.int64); mask = Tok == MASK
    P = np.zeros((n, L_, V), np.float32)
    ruido = np.exp(r.normal(0, 0.7, (n, L_, V)).astype(np.float32))
    P[mask] = (uni * ruido[mask]); P[mask] /= P[mask].sum(1, keepdims=True)
    global L
    Lsave = L; L = L_
    try:
        anterior = None
        for k in range(K):
            F = monta(Tok, mask, P); _, p, s = passe(F); P[mask] = p
            atual = (p[:, 3:].argmax(1) + 3).reshape(n, L_).ravel()
            if anterior is not None and (atual != anterior).mean() <= limiar: break
            anterior = atual
        saida = []
        for i in range(n):
            saida.append(" ".join(itos[int(w)] for w in (P[i][:, 3:].argmax(1) + 3)))
    finally:
        L = Lsave
    return saida, k + 1


def gera_commit(n=40, L_=16, passos=6, temp=0.9, topk=30, rep=0.6, seed=0):
    global L
    r = np.random.default_rng(seed)
    L = L_
    Tok = np.full((n, L_), MASK, np.int64)
    mask = Tok == MASK
    P = np.zeros((n, L_, V), np.float32)
    P[mask] = uni
    for k in range(passos):
        if not mask.any(): break
        F = monta(Tok, mask, P)
        _, p, _ = passe(F)
        pos = np.argwhere(mask)
        q = p[:, 3:].copy()
        # repetição: penaliza palavras já presentes na mesma frase
        for row, (b, _) in enumerate(pos):
            usados = Tok[b][(~mask[b])]
            for u in usados:
                if u >= 3: q[row, u - 3] *= rep
        # top-k + temperatura
        thr = -np.sort(-q, axis=1)[:, topk - 1:topk]
        q = np.where(q >= thr, q, 0)
        q = np.power(q + 1e-12, 1.0 / temp); q /= q.sum(1, keepdims=True)
        cum = q.cumsum(1); u = r.random((len(q), 1))
        escolha = (cum < u).sum(1) + 3
        conf = p[:, 3:].max(1)
        # compromete as posições mais confiantes de cada frase
        for b in range(n):
            rows = np.where(pos[:, 0] == b)[0]
            if len(rows) == 0: continue
            quant = max(1, int(np.ceil(len(rows) / (passos - k))))
            ordem = rows[np.argsort(-conf[rows])][:quant]
            for row in ordem:
                c = pos[row, 1]
                Tok[b, c] = escolha[row]; mask[b, c] = False
        # palpite macio atualizado para os que ainda estão mascarados (pos antigo)
        P[pos[:, 0], pos[:, 1]] = p
    # o que sobrou recebe o argmax
    L = 32
    saidas = []
    for b in range(n):
        saidas.append(" ".join(itos[int(w)] for w in Tok[b]))
    return saidas

def distinct2(ts):
    bi = []
    for t in ts:
        w = t.split(); bi += list(zip(w, w[1:]))
    return len(set(bi)) / max(1, len(bi))


def distinct2(ts):
    bi = []
    for t in ts:
        w = t.split(); bi += list(zip(w, w[1:]))
    return len(set(bi)) / max(1, len(bi))

if __name__ == "__main__":
    print(f"teste: {TESTE} | treino frases {len(train_ids)} | vocab {V} | K={K}", flush=True)
    treina()
    for k in [1, 2, 4, 8]:
        a, _ = avalia(passos_fixos=k)
        print(f"  passes fixos {k}: acerto mascarado {a:.1%}", flush=True)
    for lim in [0.10, 0.03]:
        a, pm = avalia(limiar=lim)
        print(f"  parada por estabilidade (<= {lim:.0%} de trocas): acerto {a:.1%} | passes médios {pm:.2f}", flush=True)
    for c in [dict(temp=1.0, topk=50, rep=1.0), dict(temp=0.9, topk=30, rep=0.6), dict(temp=0.8, topk=20, rep=0.5)]:
        am = gera_commit(**c)
        print(f"GERAÇÃO {c}: distintas {len(set(am))}/{len(am)} | distinct-2 {distinct2(am):.3f}", flush=True)
        for t in am[:12]:
            print("   >", t, flush=True)
    print("FIM", flush=True)
