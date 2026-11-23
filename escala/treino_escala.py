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



# ---------- métricas gravadas durante o treino (mesma régua de diagnosticos.py) ----------
AVALIA = int(os.environ.get("AVALIA_CADA", "50000"))
N_AVAL = int(os.environ.get("N_AVALIACAO", "300"))
ARQ_MET = os.path.join(PESOS, "metricas.jsonl")
textos_aval = []
with open(os.path.join(DIR, "teste.txt"), encoding="utf-8") as f:
    for l in f:
        if len(textos_aval) >= N_AVAL:
            break
        if l.strip():
            textos_aval.append(l.strip())


def metricas(p, passo):
    """Acerto (como o CI), acerto por posição, taxa de UNK e bits por caractere (PLL) no teste fixo."""
    Xs, ys, pos, chars, nunk, ntok = [], [], [], 0, 0, 0
    for txt in textos_aval:
        ids = [stoi.get(w, n.UNK) for w in tokens(txt)]
        if len(ids) < 3:
            continue
        chars += len(txt.replace(" ", "")); ntok += len(ids); nunk += sum(1 for i in ids if i == n.UNK)
        for j in range(len(ids)):
            w = [ids[k] if 0 <= k < len(ids) else n.PAD for k in range(j - 2, j + 3)]
            w[2] = n.MASK
            Xs.append(w); ys.append(ids[j]); pos.append(min(j, 15))
    X = np.array(Xs, dtype=np.int64); Y = np.array(ys, dtype=np.int64); P_ = np.array(pos)
    probs = []
    for i in range(0, len(X), 512):
        probs.append(n.passes(p, VAR, X[i:i + 512], K).d)
    Pr = np.concatenate(probs)
    pred = Pr.argmax(1)
    util = Y >= 3
    acc = float((pred[util] == Y[util]).mean()) if util.any() else 0.0
    por_pos = {int(k): round(float((pred[util & (P_ == k)] == Y[util & (P_ == k)]).mean()), 4)
               for k in sorted(set(P_[util].tolist()))}
    bits = float(-np.log2(np.maximum(Pr[np.arange(len(Y)), Y], 1e-12)).sum())
    m = {"passo": passo, "acerto_como_CI": round(acc, 5), "taxa_unk": round(nunk / max(ntok, 1), 5),
         "bits_por_caractere_pll": round(bits / max(chars, 1), 5), "acerto_por_posicao": por_pos,
         "frases": len(textos_aval), "t_s": round(time.time() - T_INI)}
    with open(ARQ_MET, "a", encoding="utf-8") as f:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
    log(f"métricas passo {passo}: acerto {acc:.2%} | bits/caractere {m['bits_por_caractere_pll']:.4f} | UNK {m['taxa_unk']:.2%}")
    return m

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
    if passo % AVALIA == 0:
        metricas(p, passo)
    if time.time() - T_INI > TEMPO_MAX and passo < PASSOS:
        n.salva_estado(ARQ_ESTADO, p, st, passo)
        open(ARQ_FLAG, "w").write("continua")
        log(f"tempo desta execução esgotado no passo {passo}; estado salvo")
        raise SystemExit(0)

n.salva_estado(ARQ_ESTADO, p, st, PASSOS)
open(ARQ_FLAG, "w").write("fim")
if PASSOS % AVALIA != 0:
    metricas(p, PASSOS)
acc = n.acerto(p, VAR, Xte, Yte)
log(f"acerto mascarado no teste (2+2, 1 passe por K={K}): {acc:.1%}")
np.savez(os.path.join(PESOS, f"ckpt_{VAR}.npz"), **{k: v.d for k, v in p.items()})
log("FIM")
