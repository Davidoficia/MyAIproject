"""
Demonstração do módulo de sentido SÓ com numpy (sem torch, mesmo caminho do celular).
Entrada: pesos saida_sentido/sentido.npz + itos.json, e dados de teste (teste.txt).
Saída em texto simples, para leitura humana:
  1) palavras que o modelo sugere para cada frase de teste (top 8)
  2) frases mais parecidas pelo resumo (vizinhos)
  3) mistura de duas frases de assuntos diferentes: o que ele sugere
"""
import json, os, re, sys
import numpy as np
import compat_sentido as c

PESOS = sys.argv[1]; DADOS = sys.argv[2]; SAIDA = sys.argv[3]
N = 60; TOK = re.compile(r"\w+|[^\w\s]")
p = c.carrega(f"{PESOS}/sentido.npz")
itos = json.load(open(f"{PESOS}/itos.json", encoding="utf-8"))
stoi = {w: i for i, w in enumerate(itos)}
Wout, bout = p.pop("Wout"), p.pop("bout")

textos, rows = [], []
for l in open(f"{DADOS}/teste.txt", encoding="utf-8"):
    if len(textos) >= 2000:
        break
    toks = TOK.findall(l.lower())[:N]
    ids = [stoi.get(w, 1) for w in toks]
    if len(ids) < 6:
        continue
    row = np.zeros(N, dtype=np.int64); row[:len(ids)] = ids
    textos.append(" ".join(toks)); rows.append(row)
M = np.stack(rows)
Z = np.concatenate([c.forward_np(p, M[i:i + 256]) for i in range(0, len(M), 256)])
logits = Z @ Wout + bout

def sugestoes(vetor, k=8):
    ordem = np.argsort(-vetor)
    out = []
    for i in ordem:
        if i >= 3:
            out.append(itos[i])
        if len(out) == k:
            break
    return ", ".join(out)

linhas = []
linhas.append("=== 1) O que ele sugere para cada frase de teste (frases de artigos que ele nunca viu) ===")
for i in range(0, 2000, 200):
    linhas.append(f"\nFRASE: {textos[i][:160]}")
    linhas.append(f"  sugestões: {sugestoes(logits[i])}")

linhas.append("\n\n=== 2) Frases mais parecidas pelo resumo (vizinhos) ===")
Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-9)
for i in range(0, 2000, 400):
    sim = Zn[i] @ Zn.T
    sim[i] = -2
    viz = np.argsort(-sim)[:3]
    linhas.append(f"\nFRASE: {textos[i][:160]}")
    for j in viz:
        linhas.append(f"  parecida ({sim[j]:.2f}): {textos[j][:160]}")

linhas.append("\n\n=== 3) Mistura de dois assuntos: metade de uma frase + metade de outra ===")
for i in range(0, 2000, 400):
    j = (i + 1000) % 2000
    a = [x for x in M[i] if x != 0]; b = [x for x in M[j] if x != 0]
    mix = np.zeros(N, dtype=np.int64)
    m = a[:max(1, len(a) // 2)] + b[:max(1, len(b) // 2)]
    mix[:min(len(m), N)] = m[:N]
    zm = c.forward_np(p, mix[None, :])[0]
    lm = zm @ Wout + bout
    linhas.append(f"\nFRASE A: {textos[i][:100]}")
    linhas.append(f"FRASE B (outro assunto): {textos[j][:100]}")
    linhas.append(f"  sugestões da mistura: {sugestoes(lm)}")
    linhas.append(f"  sugestões só da frase A: {sugestoes(logits[i])}")

open(SAIDA, "w", encoding="utf-8").write("\n".join(linhas) + "\n")
print("\n".join(linhas))
