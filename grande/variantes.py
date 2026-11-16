"""
Variantes do Nx: tokenizador (palavra | bpe) x embedding (congelada | treinável).
Mesmo dataset, mesmo número de passos, mesmas máscaras por palavra (RNG 7).

Métricas:
- acerto_palavra: palavra inteira certa (todos os pedaços certos). UNK conta como erro.
  É a métrica comparável entre tokenizadores.
- acerto_pedaco: acerto por pedaço mascarado. NÃO é comparável entre tokenizadores.
- acerto_palavra_sem_unk: só palavras cujo alvo não é UNK (referência com a métrica antiga).

Variáveis de ambiente: DADOS, TOKENIZADOR (palavra|bpe), EMB_TREINAVEL (0|1), PASSOS, JANELA,
NTE, NTREINO.
"""
import json, os, re, time, resource
from collections import Counter
import numpy as np

DIR = os.environ.get("DADOS", "dados_grande")
TOKN = os.environ.get("TOKENIZADOR", "palavra")
TREINAVEL = int(os.environ.get("EMB_TREINAVEL", "0"))
PASSOS = int(os.environ.get("PASSOS", "1000"))
J = int(os.environ.get("JANELA", "2"))
NTE = int(os.environ.get("NTE", "1500"))
NTREINO = int(os.environ.get("NTREINO", "300000"))
D = 64; HD = 128; B = 16; LR = 2e-3; K = 8; VOC = 8000
LP = 128   # máximo de pedaços por frase
LW = 32    # máximo de palavras por frase
PAD, UNK, MASK = 0, 1, 2
NIN = D * (2 * J + 1)
TOKR = re.compile(r"\w+|[^\w\s]")
rng = np.random.default_rng(0)
t0 = time.time()

def palavras(s):
    return TOKR.findall(s.lower())[:LW]

def linhas(caminho, limite=None):
    with open(caminho, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if limite is not None and i >= limite: break
            yield l

# ---------- tokenizador
cont = Counter()
for l in linhas(f"{DIR}/treino.txt", 1_000_000):
    cont.update(palavras(l))

if TOKN == "palavra":
    itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(VOC)]
    stoi = {w: i for i, w in enumerate(itos)}
    V = len(itos)
    def pedacos(lista):
        return [[stoi.get(w, UNK)] for w in lista]
else:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOC, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=["<PAD>", "<UNK>", "<MASK>"], show_progress=False)
    amostra = [l.strip() for l in linhas(f"{DIR}/treino.txt", NTREINO)]
    tk.train_from_iterator(amostra, trainer=trainer)
    assert tk.token_to_id("<PAD>") == 0 and tk.token_to_id("<UNK>") == 1 and tk.token_to_id("<MASK>") == 2
    V = tk.get_vocab_size()
    def pedacos(lista):
        uniq = list(set(lista))
        mapa = {w: e.ids for w, e in zip(uniq, tk.encode_batch(uniq))}
        return [mapa[w] for w in lista]
print(f"tokenizador={TOKN} | V={V} | {time.time()-t0:.0f}s", flush=True)

def monta_sentenca(sent_palavras):
    """palavras -> (ids dos pedaços, índice da palavra de cada pedaço, nº de palavras), ou None se estourar LP."""
    pcs = pedacos(sent_palavras)
    ids, wid = [], []
    for wi, p in enumerate(pcs):
        if len(ids) + len(p) > LP:
            break
        ids += p; wid += [wi] * len(p)
    if not ids: return None
    return np.array(ids, np.int64), np.array(wid, np.int64), max(wid) + 1

# ---------- dados de treino (NTREINO frases) e de teste (NTE frases)
TR = []
for l in linhas(f"{DIR}/treino.txt", NTREINO):
    ws = palavras(l)
    if len(ws) < 3: continue
    s = monta_sentenca(ws)
    if s is not None and s[2] >= 3: TR.append(s)
TE_raw = []
for l in linhas(f"{DIR}/teste.txt"):
    ws = palavras(l)
    if len(ws) < 3: continue
    s = monta_sentenca(ws)
    if s is None: continue
    TE_raw.append(s)
    if len(TE_raw) >= NTE: break
TE = TE_raw
uni = np.bincount(np.concatenate([s[0] for s in TR]), minlength=V).astype(np.float32) + 1
uni /= uni.sum()
n_te_pedacos = sum(len(s[0]) for s in TE)
n_unk_te = sum(int((s[0] == UNK).sum()) for s in TE)
print(f"treino {len(TR)} frases | teste {len(TE)} | UNK no teste (pedaços): {n_unk_te/max(1,n_te_pedacos):.1%} | {time.time()-t0:.0f}s", flush=True)

# ---------- modelo
W = {"E": rng.normal(0, 0.1, (V, D)).astype(np.float32),
     "W1": rng.normal(0, 1/np.sqrt(NIN), (NIN, HD)).astype(np.float32), "b1": np.zeros(HD, np.float32),
     "W2": rng.normal(0, 1/np.sqrt(HD), (HD, V)).astype(np.float32), "b2": np.zeros(V, np.float32),
     "ws": np.zeros(HD, np.float32), "bs": np.zeros(1, np.float32)}
W["E"][PAD] = 0
A = {k: [np.zeros_like(v), np.zeros_like(v), 0] for k, v in W.items()}

def forward(Tok, mask, P):
    E = W["E"]; Lb = Tok.shape[1]
    X = E[Tok].copy()
    X[mask] = P[mask] @ E
    Xp = np.zeros((Tok.shape[0], Lb + 2 * J, D), np.float32); Xp[:, J:J+Lb] = X
    F = np.concatenate([Xp[:, J+o:J+o+Lb] for o in range(-J, J+1)], axis=2)
    return F[mask]

def passe(F):
    H = np.tanh(F @ W["W1"] + W["b1"])
    z = H @ W["W2"] + W["b2"]
    z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    s = 1 / (1 + np.exp(-(H @ W["ws"] + W["bs"][0])))
    return H, p, s

def grad_E(Tok, mask, P, dF):
    Lb = Tok.shape[1]
    dFf = np.zeros((Tok.shape[0], Lb, NIN), np.float32); dFf[mask] = dF
    dXp = np.zeros((Tok.shape[0], Lb + 2 * J, D), np.float32)
    for bi, o in enumerate(range(-J, J+1)):
        dXp[:, J+o:J+o+Lb] += dFf[:, :, bi*D:(bi+1)*D]
    dX = dXp[:, J:J+Lb]
    dE = np.zeros_like(W["E"])
    np.add.at(dE, Tok[~mask], dX[~mask])
    dE += P[mask].T @ dX[mask]
    dE[PAD] = 0
    return dE

def adam(g):
    for k in g:
        m, v, t = A[k]; t += 1
        m[:] = 0.9*m + 0.1*g[k]; v[:] = 0.999*v + 0.001*g[k]**2
        W[k] -= LR * (m/(1-0.9**t)) / (np.sqrt(v/(1-0.999**t)) + 1e-8)
        A[k][2] = t

def lote_treino():
    idx = rng.integers(0, len(TR), B)
    Lb = max(len(TR[i][0]) for i in idx)
    Tok = np.full((B, Lb), PAD, np.int64); mask = np.zeros((B, Lb), bool)
    for r_, i in enumerate(idx):
        ids, wid, nw = TR[i]; n = len(ids)
        Tok[r_, :n] = ids
        mw = rng.random(nw) < rng.uniform(0.3, 0.6)
        if not mw.any(): mw[rng.integers(nw)] = True
        mask[r_, :n] = mw[wid]
    Y = Tok.copy()
    Tok[mask] = MASK
    return Tok, mask, Y

def treina():
    tt = time.time(); perda = 0.0; n = 0
    for passo in range(1, PASSOS + 1):
        Tok, mask, Y = lote_treino()
        P = np.zeros((B, Tok.shape[1], V), np.float32); P[mask] = uni
        t = int(rng.integers(0, K))
        for _ in range(t):
            F = forward(Tok, mask, P); _, p, _ = passe(F); P[mask] = p
        F = forward(Tok, mask, P)
        H, p, s = passe(F)
        y = Y[mask]; M = len(y)
        perda += -np.log(p[np.arange(M), y] + 1e-12).mean(); n += 1
        acerto = (p.argmax(1) == y).astype(np.float32)
        dz = p.copy(); dz[np.arange(M), y] -= 1; dz /= M
        ds = (s - acerto) / M
        dH = dz @ W["W2"].T + np.outer(ds, W["ws"])
        dpre = dH * (1 - H * H)
        g = {"W2": H.T @ dz, "b2": dz.sum(0), "W1": F.T @ dpre, "b1": dpre.sum(0),
             "ws": H.T @ ds, "bs": np.array([ds.sum()], np.float32)}
        if TREINAVEL:
            g["E"] = grad_E(Tok, mask, P, dpre @ W["W1"].T)
        adam(g)
        if passo % 200 == 0:
            print(f"  passo {passo}/{PASSOS}: perda {perda/n:.3f} | {time.time()-tt:.0f}s", flush=True)
            perda = 0.0; n = 0
    return time.time() - tt

def avalia(passos=4):
    r = np.random.default_rng(7)
    pc_ok = pc_tot = 0
    pal_ok = pal_tot = 0
    pal_ok_sem = pal_tot_sem = 0
    for b0 in range(0, len(TE), 64):
        grupo = TE[b0:b0+64]
        Lb = max(len(g[0]) for g in grupo)
        Tok = np.full((len(grupo), Lb), PAD, np.int64); Y = Tok.copy()
        Wd = np.full((len(grupo), Lb), -1, np.int64); mask = np.zeros((len(grupo), Lb), bool)
        for r_, (ids, wid, nw) in enumerate(grupo):
            n = len(ids); Tok[r_, :n] = ids; Y[r_, :n] = ids; Wd[r_, :n] = wid
            mw = r.random(nw) < 0.3
            if not mw.any(): mw[0] = True
            mask[r_, :n] = mw[wid]
        Tok[mask] = MASK
        P = np.zeros((len(grupo), Lb, V), np.float32); P[mask] = uni
        for _ in range(passos):
            F = forward(Tok, mask, P); _, p, _ = passe(F); P[mask] = p
        pred = p[:, 3:].argmax(1) + 3
        gold = Y[mask]; ok = pred == gold
        pc_ok += int(ok.sum()); pc_tot += len(ok)
        rows = np.nonzero(mask)[0]; wds = Wd[mask]
        palavras_m = {}
        for k in range(len(gold)):
            key = (int(rows[k]), int(wds[k]))
            palavras_m.setdefault(key, [True, False])
            palavras_m[key][0] &= bool(ok[k])
            palavras_m[key][1] |= bool(gold[k] == UNK)
        for certa, tem_unk in palavras_m.values():
            pal_tot += 1; pal_ok += int(certa)
            if not tem_unk:
                pal_tot_sem += 1; pal_ok_sem += int(certa)
    return {"acerto_pedaco": pc_ok / max(1, pc_tot),
            "acerto_palavra": pal_ok / max(1, pal_tot),
            "acerto_palavra_sem_unk": pal_ok_sem / max(1, pal_tot_sem)}

if __name__ == "__main__":
    t_treino = treina()
    print(f"treino terminou em {time.time()-t0:.0f}s (treino {t_treino:.0f}s)", flush=True)
    n_params = int(sum(v.size for v in W.values()))
    Tok = np.full((1000, 1), MASK, np.int64); mk = np.ones((1000, 1), bool)
    Pm = np.zeros((1000, 1, V), np.float32); Pm[:] = uni
    ts = []
    for _ in range(5):
        a = time.time(); F = forward(Tok, mk, Pm); passe(F); ts.append(time.time() - a)
    ms_1000 = 1000 * float(np.median(ts))
    m4 = avalia(4)
    m1 = avalia(1)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    resumo = {"tokenizador": TOKN, "emb_treinavel": TREINAVEL, "janela": J, "passos": PASSOS,
              "params": n_params, "V": V,
              "unk_teste_pedacos": round(n_unk_te / max(1, n_te_pedacos), 4),
              "palavra_4_passes": round(m4["acerto_palavra"], 4),
              "palavra_sem_unk_4_passes": round(m4["acerto_palavra_sem_unk"], 4),
              "pedaco_4_passes": round(m4["acerto_pedaco"], 4),
              "palavra_1_passe": round(m1["acerto_palavra"], 4),
              "treino_s": round(t_treino), "inferencia_ms_1000_pos": round(ms_1000, 2),
              "pico_ram_MB": round(rss)}
    print("RESUMO", json.dumps(resumo), flush=True)
    print("FIM", flush=True)
