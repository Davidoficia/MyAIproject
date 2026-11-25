"""Gera amostras com o checkpoint treinado (penalidade de repetição 1,5).
Variáveis (iguais às do treino): VARIANTE (padrão base), JANELA_MAX (padrão 2), RAIOS (para cresc).
Uso: python3 -I gera_amostras.py <pasta_pesos> <saida.txt> <n>"""
import json, os, sys
import numpy as np
import nx_variantes as n
n.DT = np.float32
pasta, saida, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
VAR = os.environ.get("VARIANTE", "base"); R = int(os.environ.get("JANELA_MAX", "2"))
z = np.load(f"{pasta}/ckpt_{VAR}.npz"); p = {k: n.T(z[k]) for k in z.files}
itos = json.load(open(f"{pasta}/itos.json", encoding="utf-8"))
FREQ = list(range(3, 63))
def gera(semente, rep=1.5, L=12, VAR=10):
    rng = np.random.default_rng(semente)
    seq = [int(rng.choice(FREQ)) for _ in range(L)]
    for _ in range(VAR):
        for i in rng.permutation(L):
            win = [seq[k] if 0 <= k < L else n.PAD for k in range(i - R, i + R + 1)]; win[R] = n.MASK
            P = n.passes(p, VAR, np.array([win]), n.K).d[0].astype(np.float64); P[:3] = 0
            for t in set(seq[j] for j in range(L) if j != i): P[t] /= rep
            idx = np.argsort(-P)[:20]; q = P[idx] ** (1 / 0.9); q /= q.sum()
            seq[i] = int(rng.choice(idx, p=q))
    t = " ".join(itos[i] for i in seq)
    for s in [" ,", " .", " !", " ?", " ;", " :"]: t = t.replace(s, s.strip())
    return t
with open(saida, "w", encoding="utf-8") as f:
    for s in range(N): f.write(gera(s) + "\n")
print("amostras gravadas:", saida, flush=True)
