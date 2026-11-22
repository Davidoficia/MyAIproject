"""
Checa se trechos das amostras aparecem literalmente no corpus de treino (dados_escala/treino.txt).
Janelas de 5 palavras puras (sem pontuação), comparadas sem caixa-alta, por grep -F em uma passada.
Saída: por amostra, quantas janelas aparecem no corpus e quantas vezes (top).
"""
import re, subprocess, sys, collections, os

DADOS = os.environ.get("DADOS", "dados_escala")
CORPUS = os.path.join(DADOS, "treino.txt")
ARQS = ["escala/checagem/amostras_954k.txt", "escala/checagem/amostras_3k_pt.txt"]
W = 5

def janelas(linha):
    out = []
    for seg in re.split(r"[,.;:!?()\"'\-–—«»]", linha.lower()):
        p = seg.split()
        if not all(re.fullmatch(r"[\wáéíóúâêôãõçà]+", w) for w in p):
            continue
        out += [" ".join(p[i:i+W]) for i in range(len(p)-W+1)]
    return out

amostras = {}
for a in ARQS:
    nome = "954k" if "954k" in a else "3k"
    linhas = [l.strip() for l in open(a, encoding="utf-8") if l.strip() and not l.startswith(("===", "Shell"))]
    amostras[nome] = linhas

pats = {}
for nome, linhas in amostras.items():
    for i, l in enumerate(linhas, 1):
        for j in janelas(l):
            pats.setdefault(j, set()).add((nome, i))
open("pats.txt", "w", encoding="utf-8").write("\n".join(pats) + "\n")
print(f"janelas distintas: {len(pats)}", flush=True)

conta = collections.Counter()
p = subprocess.Popen(["grep", "-i", "-o", "-F", "-f", "pats.txt", CORPUS],
                     stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
for linha in p.stdout:
    conta[" ".join(linha.lower().split())] += 1
p.wait()

for nome, linhas in amostras.items():
    tot = hit = 0
    exemplos = []
    for i, l in enumerate(linhas, 1):
        js = janelas(l)
        achou = [j for j in js if conta.get(j, 0) > 0]
        tot += len(js); hit += len(achou)
        if achou:
            exemplos.append((i, l, [(j, conta[j]) for j in achou][:3]))
    print(f"\n== {nome}: {hit} de {tot} janelas de 5 palavras aparecem no treino")
    for i, l, ex in exemplos:
        print(f"  #{i} {l}\n      -> {ex}")

print("\nTop 15 janelas mais frequentes no corpus (das amostras):")
for j, c in sorted(((j, conta[j]) for j in pats if conta.get(j)), key=lambda x: -x[1])[:15]:
    print(f"  {c:>8}  {j}")
