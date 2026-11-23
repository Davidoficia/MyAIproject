"""
Nx com janela configurável (W palavras de cada lado) e borda opcional na fronteira da frase.
  JANELA=2 BORDA=0 -> idêntico ao Nx atual (checado por CHECA=equiv contra nx_variantes.passes).
  BORDA=1          -> na 1ª posição fora da frase entra <BORDA> (id 3) em vez de <PAD>; o resto
                      continua PAD. Assim o modelo vê onde a frase acaba, sem janela fixa cega.
Variáveis: DADOS, JANELA, BORDA, PASSOS, SAIDA (json com resultado).
"""
import array, json, os, re, time
from collections import Counter
import numpy as np
import nx_variantes as n

W = int(os.environ.get("JANELA", "2"))
BORDA_ON = os.environ.get("BORDA", "0") == "1"
DIR = os.environ.get("DADOS", "dados_grande")
PASSOS = int(os.environ.get("PASSOS", "1000"))
SAIDA = os.environ.get("SAIDA", "")
B, LR, VOC, K, L = 32, 3e-3, 8000, n.K, 32
BORDA = 3
TOK = re.compile(r"\w+|[^\w\s]")


def tokens(s):
    return TOK.findall(s.lower())[:L]


def vocab():
    cont = Counter()
    with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= 1_000_000:
                break
            cont.update(tokens(l))
    itos = ["<PAD>", "<UNK>", "<MASK>", "<BORDA>"] + [w for w, _ in cont.most_common(VOC)]
    return itos, {w: i for i, w in enumerate(itos)}


MAX_LINHAS = int(os.environ.get("MAX_LINHAS", "2000000"))


def carrega(caminho, stoi):
    flat = array.array("q"); off = array.array("q", [0])
    with open(caminho, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= MAX_LINHAS:
                break
            ids = [stoi.get(w, n.UNK) for w in tokens(l)]
            if len(ids) < 3:
                continue
            flat.extend(ids); off.append(len(flat))
    return np.frombuffer(flat, dtype=np.int64), np.frombuffer(off, dtype=np.int64)


def janelas(flat, off, r, Bn):
    """Posições com alvo >= 3; janela de 2W+1 colunas, centro = MASK, fora da frase = PAD (ou BORDA)."""
    xs, ys = [], []
    faltam = Bn; N = len(flat)
    while faltam > 0:
        g = r.integers(0, N, faltam * 2)
        s = np.searchsorted(off, g, side="right") - 1
        ini, fim = off[s], off[s + 1]
        y = flat[g]
        ok = y >= 3
        g, ini, fim, y = g[ok][:faltam], ini[ok][:faltam], fim[ok][:faltam], y[ok][:faltam]
        win = np.full((len(g), 2 * W + 1), n.PAD, dtype=np.int64)
        for j, k in enumerate(range(-W, W + 1)):
            if k == 0:
                continue
            pos = g + k
            inside = (pos >= ini) & (pos < fim)
            v = np.where(inside, flat[np.clip(pos, 0, N - 1)], n.PAD)
            if BORDA_ON:
                v = np.where((pos == ini - 1) | (pos == fim), BORDA, v)
            win[:, j] = v
        win[:, W] = n.MASK
        xs.append(win); ys.append(y); faltam -= len(g)
    return np.concatenate(xs), np.concatenate(ys)


def passes_w(p, ids, K_):
    Bn = ids.shape[0]; E = p["E"]; V = E.d.shape[0]
    slots = [n.embed(E, ids[:, j]) for j in range(2 * W + 1) if j != W]
    P = n.T(np.full((Bn, V), 1.0 / V))
    for _ in range(K_):
        Xc = n.matmul(P, E)
        F = n.concat(slots[:W] + [Xc] + slots[W:])
        H = n.tanh(n.add(n.matmul(F, p["W1"]), p["b1"]))
        P = n.softmax(n.add(n.matmul(H, p["W2"]), p["b2"]))
    return P


def perda_w(p, ids, y, K_):
    P = passes_w(p, ids, K_)
    Bn, V = ids.shape[0], P.d.shape[1]
    oh = np.zeros((Bn, V)); oh[np.arange(Bn), y] = 1
    lp = n.log(n.add(P, n.T(np.full((1, 1), 1e-9))))
    return n.mul(n.soma(n.mul(n.T(oh), lp)), n.T(np.array(-1.0 / Bn)))


def acerto_w(p, X, Y, K_=K):
    ok = 0
    for i in range(0, len(X), 256):
        P = passes_w(p, X[i:i + 256], K_)
        ok += int((P.d.argmax(1) == Y[i:i + 256]).sum())
    return ok / len(X)


if os.environ.get("CHECA") == "equiv":
    rng = np.random.default_rng(0)
    p = n.init(50, "base", rng)
    X = rng.integers(3, 50, (16, 5)); X[:, 2] = n.MASK
    a = passes_w(p, X, K).d; b = n.passes(p, "base", X, K).d
    print("W=2 igual ao atual:", np.allclose(a, b, atol=1e-12), "max dif", float(np.abs(a - b).max()))
    raise SystemExit

if __name__ == "__main__":
    t0 = time.time()
    itos, stoi = vocab(); V = len(itos)
    ftr, otr = carrega(f"{DIR}/treino.txt", stoi)
    fte, ote = carrega(f"{DIR}/teste.txt", stoi)
    rng = np.random.default_rng(0)
    p = n.init(V, "base", rng)
    # janela muda o tamanho de W1: recria com nin = (2W+1)*D
    nin = (2 * W + 1) * n.D
    p["W1"] = n.T(rng.normal(0, 1 / np.sqrt(nin), (nin, n.HD)))
    st = {"t": 0, "m": {k: np.zeros_like(v.d) for k, v in p.items()},
          "v": {k: np.zeros_like(v.d) for k, v in p.items()}}
    Xte, Yte = janelas(fte, ote, np.random.default_rng(7), 2000)
    print(f"W={W} borda={BORDA_ON} V={V} passos={PASSOS}", flush=True)
    for passo in range(1, PASSOS + 1):
        Xb, yb = janelas(ftr, otr, np.random.default_rng([0, passo]), B)
        loss = perda_w(p, Xb, yb, K)
        for v in p.values(): v.g = None   # zera gradiente antes de cada passo
        n.backward(loss)
        n.adam(p, st, n.lr_em(passo, PASSOS, LR))
        if passo % max(1, PASSOS // 5) == 0 or passo == PASSOS:
            print(f"passo {passo} loss {float(loss.d):.3f} acerto_teste {acerto_w(p, Xte, Yte):.4f} t={time.time()-t0:.0f}s", flush=True)
    res = {"W": W, "borda": BORDA_ON, "passos": PASSOS, "acerto_teste": acerto_w(p, Xte, Yte)}
    print(json.dumps(res), flush=True)
    if SAIDA:
        json.dump(res, open(SAIDA, "w"))
