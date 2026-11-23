"""
Diagnósticos gravados durante o treino (pra não retreinar pra cada pergunta).
Entrada: pesos (ckpt_base.npz), vocabulário (itos.json), arquivo de teste (uma frase por linha).
Saída: JSON com acerto mascarado por posição da frase, por passes (1 e 4), taxa de UNK,
bits por caractere (comparável entre tokenizadores), e cobertura do vocabulário.
Uso: python3 -I diagnosticos.py <pasta_pesos> <teste.txt> <saida.json> [n_frases]
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
por_pos = {}          # posição na frase -> [acertos4, acertos1, n]
caracteres = 0; bits = 0.0
for f in frases:
    t = toks(f)
    ids = [stoi.get(w, n.UNK) for w in t]
    total += len(t); unk += sum(1 for i in ids if i == n.UNK)
    for j in range(len(ids)):
        if ids[j] == n.UNK or ids[j] < 3:
            continue
        win = [ids[k] if 0 <= k < len(ids) else n.PAD for k in range(j - 2, j + 3)]
        win[2] = n.MASK
        P4 = n.passes(p, "base", np.array([win]), 4).d[0]
        P1 = n.passes(p, "base", np.array([win]), 1).d[0]
        ok4 = int(np.argmax(P4[3:]) + 3 == ids[j]); ok1 = int(np.argmax(P1[3:]) + 3 == ids[j])
        r = por_pos.setdefault(min(j, 15), [0, 0, 0])
        r[0] += ok4; r[1] += ok1; r[2] += 1
        pr = max(P4[ids[j]], 1e-12) / max(P4.sum(), 1e-12)
        bits += -math.log2(pr); caracteres += len(itos[ids[j]])

res = {
    "frases": len(frases), "tokens": total,
    "taxa_unk": unk / max(total, 1),
    "acerto_4_passes": sum(v[0] for v in por_pos.values()) / max(sum(v[2] for v in por_pos.values()), 1),
    "acerto_1_passe": sum(v[1] for v in por_pos.values()) / max(sum(v[2] for v in por_pos.values()), 1),
    "acerto_por_posicao_4p": {k: round(v[0] / v[2], 4) for k, v in sorted(por_pos.items()) if v[2] > 0},
    "bits_por_caractere_aprox": bits / max(caracteres, 1),
}
json.dump(res, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ["frases", "taxa_unk", "acerto_4_passes", "acerto_1_passe", "bits_por_caractere_aprox"]}, indent=1))
