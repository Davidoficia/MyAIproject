"""
Treino em escala do Nx (mesmo modelo do nx_variantes, variante base por padrão) para o CI.
Diferença para o nx_variantes: dados em arrays int32 (sem listas Python) e janelas amostradas
direto dos tokens, para caber na RAM do runner (7 GB).
Variáveis: DADOS, PASSOS, LOTE, LR, VOCAB, VARIANTE, PESOS_DIR, SALVA_A_CADA, TEMPO_MAX_MIN.
Retoma de PESOS_DIR/estado.npz. Grava PESOS_DIR/estado.txt = "continua" ou "fim".
Uso (a partir da pasta escala/): DADOS=... PESOS_DIR=... python3 -u treino_escala.py
"""
import array, json, os, time
from collections import Counter
import numpy as np
import nx_variantes as n

n.DT = np.float32
DIR = os.environ.get("DADOS", "dados_escala")
PASSOS = int(os.environ.get("PASSOS", "477000"))
B = int(os.environ.get("LOTE", "32"))
LR = float(os.environ.get("LR", "3e-3"))
VOC = int(os.environ.get("VOCAB", "8000"))
VAR = os.environ.get("VARIANTE", "base")
PESOS = os.environ.get("PESOS_DIR", "pesos")
SALVA = int(os.environ.get("SALVA_A_CADA", "2000"))
TEMPO_MAX = float(os.environ.get("TEMPO_MAX_MIN", "1e9")) * 60
SEMENTE = 0
K = n.K
os.makedirs(PESOS, exist_ok=True)
T_INI = time.time()
ARQ_ESTADO = os.path.join(PESOS, "estado.npz")
ARQ_ITOS = os.path.join(PESOS, "itos.json")
ARQ_FLAG = os.path.join(PESOS, "estado.txt")


def log(msg):
    print(f"[{time.time()-T_INI:7.0f}s] {msg}", flush=True)


def tokens(linha):
    return n.TOK.findall(linha.lower())


# ---------- vocabulário (fixo entre rodadas: fica salvo em itos.json) ----------
if os.path.exists(ARQ_ITOS):
    itos = json.load(open(ARQ_ITOS, encoding="utf-8"))
else:
    cont = Counter()
    with open(os.path.join(DIR, "treino.txt"), encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= 1_000_000:
                break
            cont.update(tokens(l))
    itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(VOC - 3)]
    json.dump(itos, open(ARQ_ITOS, "w", encoding="utf-8"), ensure_ascii=False)
stoi = {w: i for i, w in enumerate(itos)}
V = len(itos)
log(f"vocabulário {V}")


def carrega(caminho):
    flat = array.array("i"); off = array.array("q", [0])
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            ids = [stoi.get(w, n.UNK) for w in tokens(l)]
            if len(ids) < 3:
                continue
            flat.extend(ids); off.append(len(flat))
    return np.frombuffer(flat, dtype=np.int32).copy(), np.frombuffer(off, dtype=np.int64).copy()


flat_tr, off_tr = carrega(os.path.join(DIR, "treino.txt"))
flat_te, off_te = carrega(os.path.join(DIR, "teste.txt"))
log(f"treino {len(off_tr)-1} frases, {len(flat_tr)} tokens | teste {len(off_te)-1} frases, {len(flat_te)} tokens")


def janelas(flat, off, r, Bn):
    """Amostra Bn posições com alvo >= 3, janela 2+2 limitada à frase (fora = PAD)."""
    xs, ys = [], []
    faltam = Bn
    while faltam > 0:
        g = r.integers(0, len(flat), faltam * 2)
        s = np.searchsorted(off, g, side="right") - 1
        ini, fim = off[s], off[s + 1]
        y = flat[g].astype(np.int64)
        ok = y >= 3
        g, ini, fim, y = g[ok][:faltam], ini[ok][:faltam], fim[ok][:faltam], y[ok][:faltam]
        win = np.full((len(g), 5), n.PAD, dtype=np.int64)
        for k in range(-2, 3):
            pos = g + k
            inside = (pos >= ini) & (pos < fim)
            win[:, k + 2] = np.where(inside, flat[np.clip(pos, 0, len(flat) - 1)], n.PAD)
        win[:, 2] = n.MASK
        xs.append(win); ys.append(y); faltam -= len(g)
    return np.concatenate(xs), np.concatenate(ys)


# ---------- modelo, estado e retomada ----------
p = n.init(V, VAR, np.random.default_rng(SEMENTE))
st = {"t": 0, "m": {k: np.zeros_like(v.d) for k, v in p.items()}, "v": {k: np.zeros_like(v.d) for k, v in p.items()}}
inicio = 0
if os.path.exists(ARQ_ESTADO):
    inicio = n.carrega_estado(ARQ_ESTADO, p, st)
    log(f"retomando do passo {inicio}")

Xte, Yte = janelas(flat_te, off_te, np.random.default_rng(7), 3000)
perdas = []
for passo in range(inicio + 1, PASSOS + 1):
    Xb, yb = janelas(flat_tr, off_tr, np.random.default_rng([SEMENTE, passo]), B)   # determinístico por passo
    for v in p.values(): v.g = None
    loss = n.perda(p, VAR, Xb, yb, K)
    n.backward(loss)
    n.adam(p, st, n.lr_em(passo, PASSOS, LR))
    perdas.append(float(loss.d))
    if passo % 500 == 0:
        log(f"passo {passo}/{PASSOS} | perda média (500) {np.mean(perdas[-500:]):.3f}")
    if passo % SALVA == 0:
        n.salva_estado(ARQ_ESTADO, p, st, passo)
    if time.time() - T_INI > TEMPO_MAX and passo < PASSOS:
        n.salva_estado(ARQ_ESTADO, p, st, passo)
        open(ARQ_FLAG, "w").write("continua")
        log(f"tempo desta execução esgotado no passo {passo}; estado salvo")
        raise SystemExit(0)

n.salva_estado(ARQ_ESTADO, p, st, PASSOS)
open(ARQ_FLAG, "w").write("fim")
acc = n.acerto(p, VAR, Xte, Yte)
log(f"acerto mascarado no teste (2+2, 1 passe por K={K}): {acc:.1%}")
np.savez(os.path.join(PESOS, f"ckpt_{VAR}.npz"), **{k: v.d for k, v in p.items()})
log("FIM")
