"""
Treino do módulo de sentido (encoder de compat_sentido.py) em PyTorch.
Tarefa: a partir do vetor z da frase A, prever as palavras da frase vizinha B.
Métrica: recall@50 das palavras de B no top-50, com z real x controle (z de outra frase).
Variáveis: DADOS, PASSOS, LOTE, LR, MAX_LINHAS, EVAL_CADA, SAIDA (pasta de saída).
"""
import json, math, os, re, time
from collections import Counter
import numpy as np
import torch
import compat_sentido as c

DIR = os.environ.get("DADOS", "dados_grande")
PASSOS = int(os.environ.get("PASSOS", "3000"))
LOTE = int(os.environ.get("LOTE", "64"))
LR = float(os.environ.get("LR", "3e-4"))
MAX_LINHAS = int(os.environ.get("MAX_LINHAS", "1000000"))
EVAL_CADA = int(os.environ.get("EVAL_CADA", "1000"))
SAIDA = os.environ.get("SAIDA", "saida_sentido")
VOC, N, D = 8000, 60, 64
TOK = re.compile(r"\w+|[^\w\s]")
torch.manual_seed(0)
os.makedirs(SAIDA, exist_ok=True)


def tokens(s):
    return TOK.findall(s.lower())[:N]


def vocab():
    cont = Counter()
    with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= 1_000_000:
                break
            cont.update(tokens(l))
    itos = ["<PAD>", "<UNK>", "<MASK>"] + [w for w, _ in cont.most_common(VOC)]
    return itos, {w: i for i, w in enumerate(itos)}


def frases(caminho, stoi, limite):
    out = []
    with open(caminho, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= limite:
                break
            ids = [stoi.get(w, c.PAD + 1) for w in tokens(l)]
            if len(ids) < 3:
                continue
            row = np.zeros(N, dtype=np.int64); row[:len(ids)] = ids
            out.append(row)
    return np.stack(out)


def pares(M):
    return M[:-1], M[1:]          # A = frase i, B = frase i+1 (mesmo artigo na maioria dos casos)


def alvo(B_ids, V):
    t = torch.zeros(B_ids.shape[0], V)
    t.scatter_(1, B_ids, 1.0)
    t[:, :3] = 0                  # sem PAD/UNK/MASK no alvo
    return t / t.sum(1, keepdim=True).clamp(min=1)


def avalia(tp, Wout, bout, A, Bm, ctrl=False):
    with torch.no_grad():
        acc = []
        for i in range(0, len(A), 512):
            a = torch.from_numpy(A[i:i + 512]); b = torch.from_numpy(Bm[i:i + 512])
            z = c.forward_torch(tp, a)
            if ctrl:
                z = torch.roll(z, 1, 0)
            logits = z @ Wout + bout
            t = alvo(b, V_)
            top = logits.topk(50, dim=1).indices
            hit = t.gather(1, top) > 0
            n = (t > 0).sum(1).clamp(min=1)
            acc.append((hit.sum(1).float() / n.float()).numpy())
        return float(np.concatenate(acc).mean())


if __name__ == "__main__":
    t0 = time.time()
    itos, stoi = vocab(); V_ = len(itos)
    Mtr = frases(f"{DIR}/treino.txt", stoi, MAX_LINHAS)
    Mte = frases(f"{DIR}/teste.txt", stoi, 20000)
    Atr, Btr = pares(Mtr); Ate, Bte = pares(Mte)
    print(f"V={V_} pares treino={len(Atr)} teste={len(Ate)}", flush=True)
    rng = np.random.default_rng(0)
    p0 = c.init(rng, V_, D=D, N=N, L=2, H=4, F=128)
    tp = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in p0.items()}
    Wout = torch.tensor(rng.normal(0, 1 / math.sqrt(D), (D, V_)), dtype=torch.float32, requires_grad=True)
    bout = torch.zeros(V_, requires_grad=True)
    params = list(tp.values()) + [Wout, bout]
    opt = torch.optim.Adam(params, lr=LR)
    for passo in range(1, PASSOS + 1):
        idx = np.random.default_rng([0, passo]).integers(0, len(Atr), LOTE)
        a = torch.from_numpy(Atr[idx]); b = torch.from_numpy(Btr[idx])
        z = c.forward_torch(tp, a)
        logp = torch.log_softmax(z @ Wout + bout, dim=1)
        loss = -(alvo(b, V_) * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if passo % EVAL_CADA == 0 or passo == PASSOS:
            r_real = avalia(tp, Wout.detach(), bout.detach(), Ate, Bte)
            r_ctrl = avalia(tp, Wout.detach(), bout.detach(), Ate, Bte, ctrl=True)
            print(f"passo {passo} loss {loss.item():.3f} recall@50 real {r_real:.4f} controle {r_ctrl:.4f} t={time.time()-t0:.0f}s", flush=True)
    res = {"passos": PASSOS, "recall50_real": r_real, "recall50_controle": r_ctrl, "V": V_}
    print(json.dumps(res), flush=True)
    json.dump(res, open(f"{SAIDA}/resultado.json", "w"))
    out = {k: v.detach().double().numpy() for k, v in tp.items()}
    out["Wout"] = Wout.detach().double().numpy(); out["bout"] = bout.detach().double().numpy()
    c.salva(f"{SAIDA}/sentido.npz", out)
    json.dump(itos, open(f"{SAIDA}/itos.json", "w"), ensure_ascii=False)
