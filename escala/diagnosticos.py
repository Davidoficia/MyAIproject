"""
Diagnósticos do Nx (motor nx_variantes), pra gravar durante o treino.
Uso: python3 -I diagnosticos.py <pasta_pesos> <teste.txt> <saida.json> [n_frases]

- acerto_mascarado_como_CI: argmax sobre o vocabulário inteiro (igual a nx_variantes.acerto)
- acerto_sem_especiais: argmax ignorando PAD/UNK/MASK (versão anterior, só referência)
- acerto_por_posicao: por índice do token na frase (0..15+)
- taxa_unk: fração de tokens fora do vocabulário
- bits_por_caractere_pll: pseudo-log-verossimilhança (cada token mascarado em janela 2+2,
  todos os tokens, UNK inclusive) em bits, dividida pelo número de caracteres da frase.
  É a medida comparável entre tokenizadores (mesmo texto, mesma régua).
Passes: K=4 (o que o modelo foi treinado pra usar). Também grava acerto com K=1 pra referência.
"""
import json, re, sys, math
import numpy as np
import nx_variantes as n
n.DT = np.float32

pasta, teste, saida = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
z = np.load(f"{pasta}/ckpt_base.npz"); p = {k: n.T(z[k]) for k in z.files}
itos = json.load(open(f"{pasta}/itos.json", encoding="utf-8")); stoi = {w: i for i, w in enumerate(itos)}
toks = lambda s: re.findall(r"\w+|[^\w\s]", s.lower())

frases = [l.strip() for l in open(teste, encoding="utf-8") if l.strip()][:N]
total = unk = 0
n_pos = 0; ok_ci = ok_sem = ok_k1 = 0
por_pos = {}
bits = 0.0; chars = 0
for f in frases:
    t = toks(f)
    ids = [stoi.get(w, n.UNK) for w in t]
    total += len(t); unk += sum(1 for i in ids if i == n.UNK)
    chars += len(f.replace(" ", ""))
    alvo_idx = [j for j in range(len(ids))]
    if not alvo_idx: continue
    wins = []
    for j in alvo_idx:
        w = [ids[k] if 0 <= k < len(ids) else n.PAD for k in range(j - 2, j + 3)]
        w[2] = n.MASK; wins.append(w)
    X = np.array(wins)
    P4 = n.passes(p, "base", X, 4).d
    P1 = n.passes(p, "base", X, 1).d
    for r, j in enumerate(alvo_idx):
        y = ids[j]
        if y >= 3:  # alvos especiais (PAD/UNK/MASK) não entram no acerto
            n_pos += 1
            ok_ci += int(np.argmax(P4[r]) == y)
            ok_sem += int(np.argmax(P4[r][3:]) + 3 == y)
            ok_k1 += int(np.argmax(P1[r]) == y)
            pos = min(j, 15); e = por_pos.setdefault(pos, [0, 0]); e[0] += int(np.argmax(P4[r]) == y); e[1] += 1
        bits += -math.log2(max(P4[r][y], 1e-12))
res = {
    "frases": len(frases), "tokens": total, "posicoes_avaliadas": n_pos,
    "taxa_unk": unk / max(total, 1),
    "acerto_mascarado_como_CI": ok_ci / max(n_pos, 1),
    "acerto_sem_especiais": ok_sem / max(n_pos, 1),
    "acerto_K1_referencia": ok_k1 / max(n_pos, 1),
    "acerto_por_posicao": {k: round(v[0] / v[1], 4) for k, v in sorted(por_pos.items()) if v[1]},
    "bits_por_caractere_pll": bits / max(chars, 1),
    "caracteres": chars,
}
json.dump(res, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ["frases", "taxa_unk", "acerto_mascarado_como_CI", "acerto_sem_especiais", "acerto_K1_referencia", "bits_por_caractere_pll"]}, indent=1))
