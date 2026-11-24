"""
Variantes do Nx com autograd mínimo em numpy (sem torch).
  base : MLP sobre a janela (5 posições), K passes, pesos compartilhados (como o Nx atual)
  mh   : + atenção aditiva multi-cabeça sobre as 4 posições de contexto (query = palpite do centro)
  mod  : + biblioteca de M módulos e controlador por passe, R passos em sequência
  lstm : + LSTM integrado, com estado carregado entre os passes da difusão

Uso:
  python3 nx_variantes.py gradcheck           -> confere gradientes por diferença finita
  python3 nx_variantes.py treina <variante> <passos> <wiki.txt> <nossos.jsonl>
"""
import argparse, collections, json, math, os, re, sys, time
import numpy as np

sys.setrecursionlimit(20000)
D, HD, HEADS, HA, M, R, HL, K = 32, 128, 2, 16, 3, 2, 32, 4
D = int(os.environ.get("NX_D", str(D)))   # teste de capacidade (padrão 32, igual ao Nx atual)
K = int(os.environ.get("NX_K", str(K)))   # teste de passes (padrão 4, igual ao Nx atual)
PAD, UNK, MASK = 0, 1, 2
DT = np.float64   # float32 no treino longo (main), float64 no gradcheck


# ---------------- autograd mínimo ----------------
class T:
    __slots__ = ("d", "g", "parents", "back")

    def __init__(self, d, parents=(), back=None):
        self.d = np.asarray(d, dtype=DT)
        self.g = None
        self.parents = parents
        self.back = back


def acc(t, g):
    if t.g is None:
        t.g = np.zeros_like(t.d)
    t.g += g


def unb(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def add(a, b):
    o = T(a.d + b.d, (a, b))
    def back(g):
        acc(a, unb(g, a.d.shape)); acc(b, unb(g, b.d.shape))
    o.back = back
    return o


def mul(a, b):
    o = T(a.d * b.d, (a, b))
    def back(g):
        acc(a, unb(g * b.d, a.d.shape)); acc(b, unb(g * a.d, b.d.shape))
    o.back = back
    return o


def matmul(a, b):
    o = T(a.d @ b.d, (a, b))
    def back(g):
        acc(a, g @ b.d.T); acc(b, a.d.T @ g)
    o.back = back
    return o


def tanh(a):
    y = np.tanh(a.d); o = T(y, (a,))
    def back(g): acc(a, g * (1 - y * y))
    o.back = back
    return o


def sigmoid(a):
    y = 1 / (1 + np.exp(-a.d)); o = T(y, (a,))
    def back(g): acc(a, g * y * (1 - y))
    o.back = back
    return o


def softmax(a):
    m = a.d.max(-1, keepdims=True)
    e = np.exp(a.d - m)
    y = e / e.sum(-1, keepdims=True)
    o = T(y, (a,))
    def back(g): acc(a, y * (g - (g * y).sum(-1, keepdims=True)))
    o.back = back
    return o


def log(a):
    o = T(np.log(a.d), (a,))
    def back(g): acc(a, g / a.d)
    o.back = back
    return o


def soma(a):
    o = T(a.d.sum(), (a,))
    def back(g): acc(a, np.full(a.d.shape, float(g)))
    o.back = back
    return o


def concat(xs):
    o = T(np.concatenate([x.d for x in xs], -1), tuple(xs))
    cortes = np.cumsum([x.d.shape[-1] for x in xs])[:-1]
    def back(g):
        for x, pg in zip(xs, np.split(g, cortes, axis=-1)):
            acc(x, pg)
    o.back = back
    return o


def cols(a, lo, hi):
    o = T(a.d[:, lo:hi], (a,))
    def back(g):
        z = np.zeros_like(a.d); z[:, lo:hi] = g; acc(a, z)
    o.back = back
    return o


def embed(E, idx):
    o = T(E.d[idx], (E,))
    def back(g):
        z = np.zeros_like(E.d); np.add.at(z, idx, g); acc(E, z)
    o.back = back
    return o


def backward(loss):
    ordem, vistos = [], set()
    def visita(t):
        if id(t) in vistos: return
        vistos.add(id(t))
        for p in t.parents: visita(p)
        ordem.append(t)
    visita(loss)
    loss.g = np.ones_like(loss.d)
    for t in reversed(ordem):
        if t.back is not None and t.g is not None:
            t.back(t.g)


# ---------------- modelo ----------------
def init(V, variante, rng):
    p = {}
    p["E"] = T(rng.normal(0, 0.1, (V, D))); p["E"].d[PAD] = 0
    nin = 5 * D + (HEADS * D if variante in ("mh", "tudo") else 0)
    p["W1"] = T(rng.normal(0, 1 / np.sqrt(nin), (nin, HD)))
    p["b1"] = T(np.zeros((1, HD)))
    p["W2"] = T(rng.normal(0, 1 / np.sqrt(HD), (HD, V)))
    p["b2"] = T(np.zeros((1, V)))
    if variante in ("mh", "tudo"):
        for h in range(HEADS):
            p[f"Wa{h}"] = T(rng.normal(0, 1 / np.sqrt(D), (D, HA)))
            p[f"Wb{h}"] = T(rng.normal(0, 1 / np.sqrt(D), (D, HA)))
            p[f"va{h}"] = T(rng.normal(0, 1 / np.sqrt(HA), (HA, 1)))
    if variante == "film":
        for k in range(K):
            p[f"Wg{k}"] = T(np.zeros((D, HD)))   # ganho por passe (começa em 1)
            p[f"Wb{k}"] = T(np.zeros((D, HD)))   # deslocamento por passe (começa em 0)
    if variante in ("mod", "tudo"):
        p["Wc"] = T(rng.normal(0, 0.1, (HD, M)))
        for m in range(M):
            p[f"Wm{m}"] = T(rng.normal(0, 1 / np.sqrt(HD), (HD, HD)))
            p[f"bm{m}"] = T(np.zeros((1, HD)))
    if variante in ("lstm", "tudo"):
        p["Wl"] = T(rng.normal(0, 1 / np.sqrt(D + HL), (D + HL, 4 * HL)))
        p["bl"] = T(np.zeros((1, 4 * HL)))
        p["Wh"] = T(rng.normal(0, 1 / np.sqrt(HL), (HL, HD)))
    return p


def passes(p, variante, ids, K_):
    B = ids.shape[0]; E = p["E"]; V = E.d.shape[0]
    ctx = [embed(E, ids[:, 0]), embed(E, ids[:, 1]), embed(E, ids[:, 3]), embed(E, ids[:, 4])]
    P = T(np.full((B, V), 1.0 / V))
    h = T(np.zeros((B, HL))); c = T(np.zeros((B, HL)))
    for _k in range(K_):
        Xc = matmul(P, E)                      # palpite macio do centro
        F = concat([ctx[0], ctx[1], Xc, ctx[2], ctx[3]])
        if variante in ("mh", "tudo"):
            pooled = []
            for hh in range(HEADS):
                qb = matmul(Xc, p[f"Wb{hh}"])
                sc = [matmul(tanh(add(matmul(xk, p[f"Wa{hh}"]), qb)), p[f"va{hh}"]) for xk in ctx]
                A = softmax(concat(sc))        # (B,4): peso de cada posição de contexto
                pv = None
                for j, xk in enumerate(ctx):
                    term = mul(cols(A, j, j + 1), xk)
                    pv = term if pv is None else add(pv, term)
                pooled.append(pv)
            F = concat([F] + pooled)
        pre = matmul(F, p["W1"])
        if variante in ("lstm", "tudo"):
            gates =add(matmul(concat([Xc, h]), p["Wl"]), p["bl"])
            i_ = sigmoid(cols(gates, 0, HL)); f_ = sigmoid(cols(gates, HL, 2 * HL))
            o_ = sigmoid(cols(gates, 2 * HL, 3 * HL)); g_ = tanh(cols(gates, 3 * HL, 4 * HL))
            c = add(mul(f_, c), mul(i_, g_))
            h = mul(o_, tanh(c))
            pre = add(pre, matmul(h, p["Wh"]))
        H = tanh(add(pre, p["b1"]))
        if variante == "film":
            cv = mul(add(add(add(ctx[0], ctx[1]), ctx[2]), ctx[3]), T(np.array(0.25)))
            ganho = add(matmul(cv, p[f"Wg{_k}"]), T(np.ones((1, 1))))
            H = add(mul(ganho, H), matmul(cv, p[f"Wb{_k}"]))
        if variante in ("mod", "tudo"):
            for _r in range(R):                # sequência de R passos de módulos
                pi = softmax(matmul(H, p["Wc"]))
                upd = None
                for m in range(M):
                    om = tanh(add(matmul(H, p[f"Wm{m}"]), p[f"bm{m}"]))
                    term = mul(cols(pi, m, m + 1), om)
                    upd = term if upd is None else add(upd, term)
                H = add(H, upd)
        z = add(matmul(H, p["W2"]), p["b2"])
        P = softmax(z)
    return P


def perda(p, variante, ids, y, K_):
    P = passes(p, variante, ids, K_)
    B, V = ids.shape[0], P.d.shape[1]
    oh = np.zeros((B, V)); oh[np.arange(B), y] = 1
    lp = log(add(P, T(np.full((1, 1), 1e-9))))
    return mul(soma(mul(T(oh), lp)), T(np.array(-1.0 / B)))


def adam(p, st, lr):
    st["t"] += 1
    for k, v in p.items():
        g = v.g if v.g is not None else np.zeros_like(v.d)
        st["m"][k] = 0.9 * st["m"][k] + 0.1 * g
        st["v"][k] = 0.999 * st["v"][k] + 0.001 * g * g
        mh = st["m"][k] / (1 - 0.9 ** st["t"]); vh = st["v"][k] / (1 - 0.999 ** st["t"])
        v.d -= lr * mh / (np.sqrt(vh) + 1e-8)
    p["E"].d[PAD] = 0


def acerto(p, variante, X, Y, K_=K):
    ok = 0
    for i in range(0, len(X), 256):
        P = passes(p, variante, X[i:i + 256], K_)
        ok += int((P.d.argmax(1) == Y[i:i + 256]).sum())
    return ok / len(X)


# ---------------- dados ----------------
TOK = re.compile(r"\w+|[^\w\s]")


def toks(s):
    return TOK.findall(s.lower())


def vocabulario(frases, V):
    cont = collections.Counter()
    for s in frases:
        cont.update(toks(s))
    itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(V - 3)]
    return itos, {w: i for i, w in enumerate(itos)}


def casos(frases, stoi):
    X, Y = [], []
    for s in frases:
        t = [stoi.get(w, UNK) for w in toks(s)]
        for j in range(len(t)):
            if t[j] < 3:
                continue
            win = [t[k] if 0 <= k < len(t) else PAD for k in range(j - 2, j + 3)]
            win[2] = MASK
            X.append(win); Y.append(t[j])
    return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)


# ---------------- verificação de gradiente ----------------
def gradcheck():
    for variante in ["base", "mh", "mod", "lstm", "tudo"]:
        rng = np.random.default_rng(1)
        V = 40
        p = init(V, variante, rng)
        ids = rng.integers(3, V, (4, 5)); ids[:, 2] = MASK
        y = rng.integers(3, V, 4)
        for v in p.values(): v.g = None
        backward(perda(p, variante, ids, y, 2))
        extras = {"base": [], "mh": ["Wa0", "Wb0", "va0"], "mod": ["Wc", "Wm0"], "lstm": ["Wl", "Wh"],
                  "tudo": ["Wa0", "Wb0", "va0", "Wc", "Wm0", "Wl", "Wh"]}[variante]
        erros = []
        for nome in ["E", "W1", "W2"] + extras:
            for _ in range(3):
                idx = tuple(int(rng.integers(0, s)) for s in p[nome].d.shape)
                ana = p[nome].g[idx]
                eps = 1e-5; old = p[nome].d[idx]
                p[nome].d[idx] = old + eps; l1 = perda(p, variante, ids, y, 2).d
                p[nome].d[idx] = old - eps; l2 = perda(p, variante, ids, y, 2).d
                p[nome].d[idx] = old
                num = float((l1 - l2) / (2 * eps))
                erros.append(abs(num - ana) / (abs(num) + abs(ana) + 1e-9))
        print(f"{variante:5s} erro relativo máximo = {max(erros):.2e}", flush=True)


# ---------------- treino ----------------
def lr_em(passo, passos, lr, aquec=200, piso=0.1):
    """Aquecimento linear e depois decaimento cosseno até piso*lr."""
    if passo <= aquec:
        return lr * passo / aquec
    f = (passo - aquec) / max(1, passos - aquec)
    return lr * (piso + (1 - piso) * 0.5 * (1 + math.cos(math.pi * min(1.0, f))))


def salva_estado(caminho, p, st, passo):
    tmp = caminho + ".tmp.npz"
    z = {f"p_{k}": v.d for k, v in p.items()}
    z.update({f"m_{k}": a for k, a in st["m"].items()})
    z.update({f"v_{k}": a for k, a in st["v"].items()})
    z["passo"] = np.array(passo); z["adam_t"] = np.array(st["t"])
    np.savez(tmp, **z)
    os.replace(tmp, caminho)            # troca atômica: nunca fica meio escrito


def carrega_estado(caminho, p, st):
    z = np.load(caminho)
    for k in p:
        p[k].d = z[f"p_{k}"].astype(DT)
        st["m"][k] = z[f"m_{k}"].astype(DT); st["v"][k] = z[f"v_{k}"].astype(DT)
    st["t"] = int(z["adam_t"])
    return int(z["passo"])


def treina(variante, passos, caminho_wiki, caminho_nossos, B=32, lr=3e-3, V=8000, semente=0,
           retoma=False, salva_a=2000, tempo_max=None, prefixo=None, eval_n=3000):
    t0 = time.time()
    prefixo = prefixo or variante
    arq_estado, arq_itos = f"estado_{prefixo}.npz", f"itos_{prefixo}.json"
    if os.environ.get("SPLIT_TR"):      # split fixo por artigo (arquivos prontos)
        tr = [l.strip() for l in open(os.environ["SPLIT_TR"], encoding="utf-8") if l.strip()]
        te = [l.strip() for l in open(os.environ["SPLIT_TE"], encoding="utf-8") if l.strip()]
    else:
        frases = [l.strip() for l in open(caminho_wiki, encoding="utf-8") if l.strip()]
        tr = [s for i, s in enumerate(frases) if i % 5 != 0]
        te = [s for i, s in enumerate(frases) if i % 5 == 0]
    if retoma and os.path.exists(arq_itos):          # vocabulário tem de ser o mesmo da primeira execução
        itos = json.load(open(arq_itos, encoding="utf-8")); stoi = {w: i for i, w in enumerate(itos)}
    else:
        itos, stoi = vocabulario(tr, V)
        json.dump(itos, open(arq_itos, "w", encoding="utf-8"), ensure_ascii=False)
    Xtr, Ytr = casos(tr, stoi)
    Xte, Yte = casos(te, stoi)
    r = np.random.default_rng(7)
    sel = r.choice(len(Xte), min(eval_n, len(Xte)), replace=False)
    Xte, Yte = Xte[sel], Yte[sel]
    nossos = [json.loads(l) for l in open(caminho_nossos, encoding="utf-8")]
    frases_n = []
    for row in nossos:
        for campo in ("q", "a"):
            frases_n += [x for x in re.split(r"[.!?\n]+", row[campo]) if x.strip()]
    Xn, Yn = casos(frases_n, stoi)
    sel = r.choice(len(Xn), min(eval_n, len(Xn)), replace=False)
    Xn, Yn = Xn[sel], Yn[sel]
    print(f"variante={variante} | dtype={np.dtype(DT).name} | treino {len(Xtr)} casos | wiki-teste {len(Xte)} | nosso {len(Xn)} | {time.time()-t0:.0f}s", flush=True)

    p = init(len(itos), variante, np.random.default_rng(semente))
    st = {"t": 0, "m": {k: np.zeros_like(v.d) for k, v in p.items()}, "v": {k: np.zeros_like(v.d) for k, v in p.items()}}
    inicio = 0
    if retoma and os.path.exists(arq_estado):
        inicio = carrega_estado(arq_estado, p, st)
        print(f"  retomando do passo {inicio}", flush=True)
    perdas = []
    parou_cedo = False
    for passo in range(inicio + 1, passos + 1):
        rng = np.random.default_rng([semente, passo])     # lote depende só de (semente, passo): retomada exata
        idx = rng.integers(0, len(Xtr), B)
        for v in p.values(): v.g = None
        loss = perda(p, variante, Xtr[idx], Ytr[idx], K)
        backward(loss)
        adam(p, st, lr_em(passo, passos, lr))
        perdas.append(float(loss.d))
        if passo % 200 == 0:
            print(f"  passo {passo} | perda média últimos 200 = {np.mean(perdas[-200:]):.3f} | {time.time()-t0:.0f}s", flush=True)
        if passo % salva_a == 0:
            salva_estado(arq_estado, p, st, passo)
        if tempo_max and time.time() - t0 > tempo_max and passo < passos:
            salva_estado(arq_estado, p, st, passo)
            print(f"PARCIAL passo={passo}/{passos} (tempo máximo atingido; rode de novo com --retoma)", flush=True)
            parou_cedo = True
            break
    if parou_cedo:
        return None
    salva_estado(arq_estado, p, st, passos)
    res = {
        "variante": variante, "passos": passos, "dtype": np.dtype(DT).name,
        "params": int(sum(v.d.size for v in p.values())),
        "acerto_wiki": acerto(p, variante, Xte, Yte),
        "acerto_nosso": acerto(p, variante, Xn, Yn),
        "tempo_s": round(time.time() - t0),
    }
    print("RESUMO " + json.dumps(res), flush=True)
    np.savez(f"ckpt_{prefixo}.npz", **{k: v.d for k, v in p.items()})
    return res


if __name__ == "__main__":
    if sys.argv[1] == "gradcheck":
        gradcheck()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("variante"); ap.add_argument("passos", type=int)
        ap.add_argument("wiki"); ap.add_argument("nossos")
        ap.add_argument("--retoma", action="store_true")
        ap.add_argument("--float64", action="store_true")
        ap.add_argument("--salva-a", type=int, default=2000)
        ap.add_argument("--tempo-max", type=float, default=None, help="segundos; salva e para (para CI de 6h)")
        ap.add_argument("--prefixo", default=None)
        ap.add_argument("--eval-n", type=int, default=3000)
        a = ap.parse_args()
        DT = np.float64 if a.float64 else np.float32
        treina(a.variante, a.passos, a.wiki, a.nossos, retoma=a.retoma, salva_a=a.salva_a,
               tempo_max=a.tempo_max, prefixo=a.prefixo, eval_n=a.eval_n)
