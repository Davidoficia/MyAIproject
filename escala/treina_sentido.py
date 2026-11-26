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
POOL = os.environ.get("POOL", "mean")     # mean | att (resumo da frase)
CTX = int(os.environ.get("CTX", "1"))        # frases de contexto: 1 ou 3
WARMUP = int(os.environ.get("WARMUP", "0"))  # passos de aquecimento da taxa (0 = sem)
ARM = os.environ.get("ARM", "solo")          # nome do braço (para o medidor)
VOC, N, D = 8000, 60, 64
NA = N * CTX   # largura da entrada (contexto)
INICIO = os.environ.get("INICIO", "")        # pesos do elo anterior (cadeia); vazio = do zero
PASSO_BASE = int(os.environ.get("PASSO_BASE", "0"))  # passos já feitos em elos anteriores
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
    if CTX == 1:
        return M[:-1], M[1:]      # A = frase i, B = frase i+1 (mesmo artigo na maioria dos casos)
    idx = np.arange(CTX - 1, len(M) - 1)   # A = frases i-2..i (CTX=3), B = frase i+1
    A = np.concatenate([M[idx - k] for k in range(CTX - 1, -1, -1)], axis=1)
    return A, M[idx + 1]


def fwd(tp, ids):
    return c.forward_torch(tp, ids, pool=POOL)


def alvo(B_ids, V):
    t = torch.zeros(B_ids.shape[0], V)
    t.scatter_(1, B_ids, 1.0)
    t[:, :3] = 0                  # sem PAD/UNK/MASK no alvo
    return t / t.sum(1, keepdim=True).clamp(min=1)


TOP_FREQ = 200   # métrica só em palavras fora das 200 mais frequentes (senão "de/a/o" dominam)


def avalia(tp, Wout, bout, A, Bm, ctrl=False, prior=None):
    """ctrl=True: usa o z de uma frase ~1000 posições antes (outro artigo, pois o teste está em ordem de artigo)."""
    with torch.no_grad():
        if prior is None:
            Z = torch.cat([fwd(tp, torch.from_numpy(A[i:i + 512])) for i in range(0, len(A), 512)])
            if ctrl:
                Z = torch.roll(Z, 1000, 0)
            LOG_ALL = Z @ Wout + bout
        acc = []
        for i in range(0, len(A), 512):
            b = torch.from_numpy(Bm[i:i + 512])
            if prior is not None:
                logits = prior.expand(b.shape[0], -1).clone()
            else:
                logits = LOG_ALL[i:i + 512].clone()
            logits[:, :TOP_FREQ] = -1e9
            t = alvo(b, V_)
            t[:, :TOP_FREQ] = 0
            top = logits.topk(50, dim=1).indices
            hit = t.gather(1, top) > 0
            n = (t > 0).sum(1).clamp(min=1)
            acc.append((hit.sum(1).float() / n.float()).numpy())
        return float(np.concatenate(acc).mean())


def perturba(M, seed=1):
    """Mudança leve e reproduzível da frase de entrada: tira ~20% das palavras e troca ~10% dos vizinhos."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(M)
    for i, row in enumerate(M):
        ids = [int(x) for x in row if x != 0]
        if len(ids) > 3:
            keep = [x for x in ids if rng.random() > 0.2]
            if len(keep) < 3:
                keep = ids[:3]
            j = 0
            while j < len(keep) - 1:
                if rng.random() < 0.1:
                    keep[j], keep[j + 1] = keep[j + 1], keep[j]
                    j += 2
                else:
                    j += 1
            ids = keep
        out[i, :min(len(ids), M.shape[1])] = ids[:M.shape[1]]
    return out


if __name__ == "__main__":
    t0 = time.time()
    itos, stoi = vocab(); V_ = len(itos)
    Mtr = frases(f"{DIR}/treino.txt", stoi, MAX_LINHAS)
    Mte = frases(f"{DIR}/teste.txt", stoi, 20000)
    Atr, Btr = pares(Mtr); Ate, Bte = pares(Mte)
    print(f"V={V_} pares treino={len(Atr)} teste={len(Ate)}", flush=True)
    rng = np.random.default_rng(0)
    p0 = c.init(rng, V_, D=D, N=NA, L=2, H=4, F=128)
    if POOL == "att":
        p0["wq"] = rng.normal(0, 1 / math.sqrt(D), (D,))
    tp = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in p0.items()}
    Wout = torch.tensor(rng.normal(0, 1 / math.sqrt(D), (D, V_)), dtype=torch.float32, requires_grad=True)
    bout = torch.zeros(V_, requires_grad=True)
    cont = np.bincount(Btr.ravel(), minlength=V_).astype(np.float64) + 1
    prior = torch.tensor(np.log(cont / cont.sum()), dtype=torch.float32).unsqueeze(0)
    r_prior = avalia(None, None, None, Ate, Bte, prior=prior)
    print(f"baseline prior (sem sentido) recall@50 = {r_prior:.4f}", flush=True)
    if INICIO:
        z_ini = np.load(INICIO)
        with torch.no_grad():
            for k in tp:
                tp[k].copy_(torch.tensor(z_ini[k], dtype=torch.float32))
            Wout.copy_(torch.tensor(z_ini["Wout"], dtype=torch.float32))
            bout.copy_(torch.tensor(z_ini["bout"], dtype=torch.float32))
        print(f"pesos do elo anterior carregados: {INICIO}", flush=True)
    params = list(tp.values()) + [Wout, bout]
    opt = torch.optim.Adam(params, lr=LR)
    for passo in range(1, PASSOS + 1):
        idx = np.random.default_rng([0, passo]).integers(0, len(Atr), LOTE)
        a = torch.from_numpy(Atr[idx]); b = torch.from_numpy(Btr[idx])
        z = fwd(tp, a)
        logp = torch.log_softmax(z @ Wout + bout, dim=1)
        loss = -(alvo(b, V_) * logp).sum(1).mean()
        if WARMUP and not INICIO:
            for g_ in opt.param_groups:
                g_["lr"] = LR * min(1.0, passo / WARMUP)
        opt.zero_grad(); loss.backward(); opt.step()
        if passo % EVAL_CADA == 0 or passo == PASSOS:
            r_real = avalia(tp, Wout.detach(), bout.detach(), Ate, Bte)
            r_ctrl = avalia(tp, Wout.detach(), bout.detach(), Ate, Bte, ctrl=True)
            print(f"passo {passo} loss {loss.item():.3f} recall@50 real {r_real:.4f} controle {r_ctrl:.4f} prior {r_prior:.4f} t={time.time()-t0:.0f}s", flush=True)
            with open(os.path.join(SAIDA, "historico.jsonl"), "a") as hf:
                hf.write(json.dumps({"passo": PASSO_BASE + passo, "real": r_real, "controle": r_ctrl}) + "\n")
    Ate_p = perturba(Ate)
    r_pert = avalia(tp, Wout.detach(), bout.detach(), Ate_p, Bte)
    print(f"frase alterada (mesmo tema, palavras tiradas/trocadas): recall@50 {r_pert:.4f} (original {r_real:.4f}, outro assunto {r_ctrl:.4f})", flush=True)
    # ambiguidade: frases curtas (menos de 8 palavras), em que o assunto é difícil de inferir
    tam = (Ate != 0).sum(1)
    curtas = tam < 8
    r_curtas = avalia(tp, Wout.detach(), bout.detach(), Ate[curtas], Bte[curtas]) if curtas.sum() > 50 else float("nan")
    # vários temas: metade da frase A do artigo i + metade da frase de outro artigo (i+1000); alvo = frase B do artigo i
    mistura = np.zeros_like(Ate)
    for i in range(len(Ate)):
        j = (i + 1000) % len(Ate)
        a = [x for x in Ate[i] if x != 0]; b = [x for x in Ate[j] if x != 0]
        m = a[:max(1, len(a) // 2)] + b[:max(1, len(b) // 2)]
        mistura[i, :min(len(m), Ate.shape[1])] = m[:Ate.shape[1]]
    r_mistura = avalia(tp, Wout.detach(), bout.detach(), mistura, Bte)
    print(f"ambiguidade (frases curtas, n={int(curtas.sum())}): recall@50 {r_curtas:.4f}", flush=True)
    print(f"vários temas (mistura de dois artigos): recall@50 {r_mistura:.4f} (original {r_real:.4f})", flush=True)
    res = {"arm": ARM, "pool": POOL, "ctx": CTX, "warmup": WARMUP, "lote": LOTE, "passos": PASSOS, "recall50_real": r_real, "recall50_controle": r_ctrl, "recall50_prior": r_prior, "recall50_alterada": r_pert, "recall50_curtas": r_curtas, "recall50_mistura": r_mistura, "V": V_}
    print(json.dumps(res), flush=True)
    json.dump(res, open(f"{SAIDA}/resultado.json", "w"))
    out = {k: v.detach().double().numpy() for k, v in tp.items()}
    out["Wout"] = Wout.detach().double().numpy(); out["bout"] = bout.detach().double().numpy()
    c.salva(f"{SAIDA}/sentido.npz", out)
    json.dump(itos, open(f"{SAIDA}/itos.json", "w"), ensure_ascii=False)
