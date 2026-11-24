"""
Teste de subpalavras no módulo de sentido.
Mudança ÚNICA em relação a treina_sentido.py: a ENTRADA usa 16 mil palavras + pedaços (subpalavras)
para palavras raras e nomes, em vez de UNK. O ALVO continua sendo as 8 mil palavras do baseline,
então a métrica de tema (recall@50 fora das 200 mais comuns) é comparável.
"""
import json, math, os, re, time
from collections import Counter
import numpy as np
import torch
import compat_sentido as c

DIR = os.environ.get("DADOS", "dados_grande")
PASSOS = int(os.environ.get("PASSOS", "30000"))
LOTE = int(os.environ.get("LOTE", "64"))
LR = float(os.environ.get("LR", "3e-4"))
MAX_LINHAS = int(os.environ.get("MAX_LINHAS", "1000000"))
EVAL_CADA = int(os.environ.get("EVAL_CADA", "5000"))
SAIDA = os.environ.get("SAIDA", "saida_sentido_sub")
VOC_OUT, VOC_IN, N_IN, N_OUT, D = 8000, 16000, 120, 60, 64
TOP_FREQ = 200
PIECES = 2000
TOK = re.compile(r"\w+|[^\w\s]")
torch.manual_seed(0)
os.makedirs(SAIDA, exist_ok=True)


def constroi():
    cont = Counter()
    with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= 1_000_000:
                break
            cont.update(TOK.findall(l.lower()))
    mais = [w for w, _ in cont.most_common(VOC_IN)]
    marcados = set(mais)
    out_itos = ["<PAD>", "<UNK>", "<MASK>"] + mais[:VOC_OUT]
    # subpalavras: n-gramas (1 a 4 letras) das palavras que ficaram fora da lista
    grams = Counter()
    for w, n in cont.items():
        if w in marcados or not w.isalpha():
            continue
        for L in range(1, 5):
            for i in range(len(w) - L + 1):
                grams[w[i:i + L]] += n
    pedacos = [g for g, _ in grams.most_common(PIECES)]
    for ch in set("".join(grams.keys())):   # garante que todo caractere tem pedaço
        if ch not in pedacos:
            pedacos.append(ch)
    in_itos = ["<PAD>", "<UNK>", "<MASK>"] + mais
    for p_ in pedacos:
        in_itos.append(p_)          # pedaço no início de palavra
        in_itos.append("##" + p_)   # pedaço no meio/fim
    return out_itos, in_itos


def ids_entrada(toks, sin):
    ids = []
    for w in toks:
        if w in sin:
            ids.append(sin[w]); continue
        i = 0
        while i < len(w):
            achou = False
            for L in range(min(4, len(w) - i), 0, -1):
                p_ = w[i:i + L]
                chave = p_ if i == 0 else "##" + p_
                if chave in sin:
                    ids.append(sin[chave]); i += L; achou = True; break
            if not achou:
                ids.append(1); i += 1
    return ids[:N_IN]


def carrega(caminho, sin, sout, limite):
    A, B = [], []
    with open(caminho, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= limite:
                break
            toks = TOK.findall(l.lower())
            ia = ids_entrada(toks, sin)
            ib = [sout.get(w, 1) for w in toks][:N_OUT]
            if len(ia) < 3 or len(ib) < 3:
                continue
            ra = np.zeros(N_IN, dtype=np.int64); ra[:len(ia)] = ia
            rb = np.zeros(N_OUT, dtype=np.int64); rb[:len(ib)] = ib
            A.append(ra); B.append(rb)
    return np.stack(A), np.stack(B)


def alvo(B_ids, V):
    t = torch.zeros(B_ids.shape[0], V)
    t.scatter_(1, B_ids, 1.0)
    t[:, :3] = 0
    return t / t.sum(1, keepdim=True).clamp(min=1)


def perturba(M, seed=1):
    rng = np.random.default_rng(seed)
    out = np.zeros_like(M)
    for i, row in enumerate(M):
        ids = [int(x) for x in row if x != 0]
        if len(ids) > 3:
            keep = [x for x in ids if rng.random() > 0.2] or ids[:3]
            j = 0
            while j < len(keep) - 1:
                if rng.random() < 0.1:
                    keep[j], keep[j + 1] = keep[j + 1], keep[j]; j += 2
                else:
                    j += 1
            ids = keep
        out[i, :min(len(ids), N_IN)] = ids[:N_IN]
    return out


def avalia(tp, Wout, bout, A, Bm, V, ctrl=False, prior=None):
    with torch.no_grad():
        if prior is None:
            Z = torch.cat([c.forward_torch(tp, torch.from_numpy(A[i:i + 512])) for i in range(0, len(A), 512)])
            if ctrl:
                Z = torch.roll(Z, 1000, 0)
            LOG = Z @ Wout + bout
        acc = []
        for i in range(0, len(A), 512):
            b = torch.from_numpy(Bm[i:i + 512])
            logits = prior.expand(b.shape[0], -1).clone() if prior is not None else LOG[i:i + 512].clone()
            logits[:, :TOP_FREQ] = -1e9
            t = alvo(b, V)
            t[:, :TOP_FREQ] = 0
            top = logits.topk(50, dim=1).indices
            hit = t.gather(1, top) > 0
            n = (t > 0).sum(1).clamp(min=1)
            acc.append((hit.sum(1).float() / n.float()).numpy())
        return float(np.concatenate(acc).mean())


if __name__ == "__main__":
    t0 = time.time()
    out_itos, in_itos = constroi()
    sout = {w: i for i, w in enumerate(out_itos)}
    sin = {w: i for i, w in enumerate(in_itos)}
    V_OUT, V_IN = len(out_itos), len(in_itos)
    Atr, Btr = carrega(f"{DIR}/treino.txt", sin, sout, MAX_LINHAS)
    Ate, Bte = carrega(f"{DIR}/teste.txt", sin, sout, 20000)
    Atr, Btr = Atr[:-1], Btr[1:]; Ate, Bte = Ate[:-1], Bte[1:]
    unk_in = float((Ate == 1).sum() / max(1, (Ate != 0).sum()))
    print(f"V_entrada={V_IN} V_alvo={V_OUT} pares treino={len(Atr)} teste={len(Ate)} UNK na entrada (teste)={unk_in*100:.2f}%", flush=True)
    json.dump(in_itos, open(f"{SAIDA}/in_itos.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(out_itos, open(f"{SAIDA}/out_itos.json", "w", encoding="utf-8"), ensure_ascii=False)
    rng = np.random.default_rng(0)
    p0 = c.init(rng, V_IN, D=D, N=N_IN, L=2, H=4, F=128)
    tp = {k: torch.tensor(v, dtype=torch.float32, requires_grad=True) for k, v in p0.items()}
    Wout = torch.tensor(rng.normal(0, 1 / math.sqrt(D), (D, V_OUT)), dtype=torch.float32, requires_grad=True)
    bout = torch.zeros(V_OUT, requires_grad=True)
    opt = torch.optim.Adam(list(tp.values()) + [Wout, bout], lr=LR)
    cont = np.bincount(Btr.ravel(), minlength=V_OUT).astype(np.float64) + 1
    prior = torch.tensor(np.log(cont / cont.sum()), dtype=torch.float32).unsqueeze(0)
    r_prior = avalia(None, None, None, Ate, Bte, V_OUT, prior=prior)
    print(f"baseline prior recall@50 = {r_prior:.4f}", flush=True)
    for passo in range(1, PASSOS + 1):
        idx = np.random.default_rng([0, passo]).integers(0, len(Atr), LOTE)
        z = c.forward_torch(tp, torch.from_numpy(Atr[idx]))
        logp = torch.log_softmax(z @ Wout + bout, dim=1)
        loss = -(alvo(torch.from_numpy(Btr[idx]), V_OUT) * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if passo % EVAL_CADA == 0 or passo == PASSOS:
            Wd, bd = Wout.detach(), bout.detach()
            r_real = avalia(tp, Wd, bd, Ate, Bte, V_OUT)
            r_ctrl = avalia(tp, Wd, bd, Ate, Bte, V_OUT, ctrl=True)
            print(f"passo {passo} loss {loss.item():.3f} recall@50 real {r_real:.4f} controle {r_ctrl:.4f} prior {r_prior:.4f} t={time.time()-t0:.0f}s", flush=True)
    Wd, bd = Wout.detach(), bout.detach()
    Ate_p = perturba(Ate)
    r_pert = avalia(tp, Wd, bd, Ate_p, Bte, V_OUT)
    tam = (Ate != 0).sum(1); curtas = tam < 8
    r_curtas = avalia(tp, Wd, bd, Ate[curtas], Bte[curtas], V_OUT) if curtas.sum() > 50 else float("nan")
    mistura = np.zeros_like(Ate)
    for i in range(len(Ate)):
        j = (i + 1000) % len(Ate)
        a = [x for x in Ate[i] if x != 0]; b = [x for x in Ate[j] if x != 0]
        m = a[:max(1, len(a) // 2)] + b[:max(1, len(b) // 2)]
        mistura[i, :min(len(m), N_IN)] = m[:N_IN]
    r_mistura = avalia(tp, Wd, bd, mistura, Bte, V_OUT)
    res = {"passos": PASSOS, "V_entrada": V_IN, "unk_entrada": unk_in, "recall50_real": r_real, "recall50_controle": r_ctrl,
           "recall50_prior": r_prior, "recall50_alterada": r_pert, "recall50_curtas": r_curtas, "recall50_mistura": r_mistura}
    print("FINAL " + json.dumps(res), flush=True)
    json.dump(res, open(f"{SAIDA}/resultado.json", "w"))
    out = {k: v.detach().double().numpy() for k, v in tp.items()}
    out["Wout"] = Wd.double().numpy(); out["bout"] = bd.double().numpy()
    c.salva(f"{SAIDA}/sentido.npz", out)
