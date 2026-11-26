"""Avalia o modelo treinado em blocos de contas ligadas por < e > (dados_bloco).
Para cada linha de teste, mascara um de cada vez:
  - cada casa (C, D, U) do resultado de cada conta da linha;
  - cada separador < ou >.
Mede a % de acerto por tipo de posição. Atenção: a janela é simétrica, então ao
prever um separador o modelo vê também a conta seguinte (que determina a direção).
Uso: python3 avalia_bloco.py <pasta_pesos> <var> <R> <teste.txt>"""
import json
import os
import sys

import numpy as np

import nx_variantes as n

n.DT = np.float32
pasta, var, R, arq = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
z = np.load(f"{pasta}/ckpt_{var}.npz")
p = {k: n.T(z[k]) for k in z.files}
itos = json.load(open(f"{pasta}/itos.json", encoding="utf-8"))
stoi = {w: i for i, w in enumerate(itos)}
linhas = [l.strip() for l in open(arq, encoding="utf-8") if l.strip()]

acertos = {"casa_resultado": [0, 0], "separador": [0, 0]}
blocos_certos = 0
for l in linhas:
    toks = n.TOK.findall(l.lower())
    ids = [stoi.get(w, n.UNK) for w in toks]
    # casas do resultado: dígitos cujo operador anterior (andando para trás) é '='
    alvos = []
    for i, t in enumerate(toks):
        if t in ("<", ">"):
            alvos.append((i, "separador"))
        elif t.isdigit() and i >= 1:
            # dígito que vem depois de '=' em alguma parte da conta
            j = i - 1
            while j >= 0 and toks[j] not in ("=", "<", ">", "+", "-"):
                j -= 1
            if j >= 0 and toks[j] == "=":
                alvos.append((i, "casa_resultado"))
    certas_linha = 0
    for i, tipo in alvos:
        win = [ids[k] if 0 <= k < len(ids) else n.PAD for k in range(i - R, i + R + 1)]
        win[R] = n.MASK
        P = n.passes(p, var, np.array([win]), n.K).d[0].astype(np.float64)
        P[:3] = 0
        ok = itos[int(np.argmax(P))] == toks[i]
        acertos[tipo][0] += int(ok)
        acertos[tipo][1] += 1
        certas_linha += int(ok)
    blocos_certos += int(certas_linha == len(alvos))

print(f"{os.path.basename(pasta)} | linhas de teste: {len(linhas)} | blocos com tudo certo: {100*blocos_certos/len(linhas):.1f}%")
for k, (a, t) in acertos.items():
    if t:
        print(f"  {k}: {100*a/t:.1f}% ({a}/{t})")
