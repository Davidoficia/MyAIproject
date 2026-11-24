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

TOP_FREQ = 200   # esconde as 200 palavras mais comuns (de, a, e, vírgula...) para mostrar o assunto


def sugestoes(vetor, k=8):
    ordem = np.argsort(-vetor)
    out = []
    for i in ordem:
        if TOP_FREQ <= i and i >= 3:
            out.append(itos[i])
        if len(out) == k:
            break
    return ", ".join(out)

linhas = []
linhas.append("=== 1) Palavras de assunto que ele sugere (sem as 200 mais comuns) ===")
for i in range(0, 2000, 200):
    linhas.append(f"\nFRASE: {textos[i][:160]}")
    linhas.append(f"  sugestões de assunto: {sugestoes(logits[i])}")

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


MESES = {"janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"}
def ehdata(i):
    w = itos[i]
    return w.isdigit() or w in MESES

def limpa_datas(row):
    out = np.zeros_like(row)
    keep = [x for x in row if x != 0 and not ehdata(int(x))]
    out[:len(keep)] = keep
    return out

def taxa_datas(logits_mat):
    tot = 0; dat = 0
    for v in logits_mat:
        for i in sugestoes_ids(v):
            tot += 1; dat += ehdata(i)
    return dat / max(tot, 1)

def sugestoes_ids(vetor, k=8):
    ordem = np.argsort(-vetor); out = []
    for i in ordem:
        if TOP_FREQ <= i and i >= 3:
            out.append(int(i))
        if len(out) == k:
            break
    return out

linhas.append("\n\n=== 4) Teste: se tirar números e nomes de meses da frase, o que muda? ===")
Mlimpo = np.stack([limpa_datas(r) for r in M])
Zl = np.concatenate([c.forward_np(p, Mlimpo[i:i + 256]) for i in range(0, len(Mlimpo), 256)])
logits_l = Zl @ Wout + bout
t_orig = taxa_datas(logits)
t_limpo = taxa_datas(logits_l)
linhas.append(f"  Sugestões que são datas/meses, frase original: {t_orig*100:.1f}%")
linhas.append(f"  Sugestões que são datas/meses, frase SEM datas: {t_limpo*100:.1f}%")
for i in range(0, 2000, 500):
    if any(ehdata(int(x)) for x in M[i] if x != 0):
        linhas.append(f"\nFRASE (sem datas): {' '.join(itos[int(x)] for x in Mlimpo[i] if x != 0)[:140]}")
        linhas.append(f"  sugestões antes: {sugestoes(logits[i])}")
        linhas.append(f"  sugestões depois: {sugestoes(logits_l[i])}")

open(SAIDA, "w", encoding="utf-8").write("\n".join(linhas) + "\n")
print("\n".join(linhas))
