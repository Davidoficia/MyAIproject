"""Mostra o que o motor de sentido prevê: para algumas frases do teste, o contexto (3 frases),
as palavras reais da frase seguinte (fora das 200 mais comuns) e as 15 previstas. Escreve no resumo do CI.
Uso (variáveis: SAIDA = pasta com sentido.npz e itos.json; DADOS; POOL=att; CTX=3)."""
import json, os
import numpy as np
import torch
import treina_sentido as ts

SAIDA = os.environ["SAIDA"]
itos = json.load(open(f"{SAIDA}/itos.json", encoding="utf-8"))
stoi = {w: i for i, w in enumerate(itos)}
z = np.load(f"{SAIDA}/sentido.npz")
tp = {k: torch.tensor(z[k], dtype=torch.float32) for k in z.files if k not in ("Wout", "bout")}
Wout = torch.tensor(z["Wout"], dtype=torch.float32)
bout = torch.tensor(z["bout"], dtype=torch.float32)

if os.environ.get("TEXTOS"):
    # texto próprio (uma frase por linha, em ordem): todas as janelas de 3 frases
    rows = []
    for l in open(os.environ["TEXTOS"], encoding="utf-8"):
        if l.strip():
            ids = [stoi.get(w, ts.c.PAD + 1) for w in ts.tokens(l)]
            r = np.zeros(ts.N, dtype=np.int64); r[:min(len(ids), ts.N)] = ids[:ts.N]; rows.append(r)
    A, B = ts.pares(np.stack(rows))
    idx = np.arange(len(A))
else:
    M = ts.frases(os.path.join(ts.DIR, "teste.txt"), stoi, 3000)
    A, B = ts.pares(M)
    idx = np.random.default_rng(0).choice(len(A), 6, replace=False)
with torch.no_grad():
    logits = (ts.fwd(tp, torch.from_numpy(A[idx])) @ Wout + bout).numpy()

out = ["## O que o sentido prevê (elo)", "", "Não gera frases: mostra as palavras que ele acha que vêm a seguir.", f"Fonte: {os.environ.get('TEXTOS', 'teste (Wikipédia, artigos fora do treino)')}", ""]
for k, i in enumerate(idx):
    ctx = " ".join(itos[t] for t in A[i] if t > 2)
    real = sorted({itos[t] for t in B[i] if t > 2 and t >= 203})
    lg = logits[k].copy()
    lg[:203] = -1e9                      # ignora as 200 mais comuns (senão "de/a/o" dominam)
    prev = [itos[t] for t in np.argsort(-lg)[:15]]
    acertos = len(set(prev) & set(real))
    out += [
        f"### Exemplo {k + 1}",
        f"**Contexto (3 frases):** {ctx[:400]}",
        f"**Palavras reais da frase seguinte:** {', '.join(real)[:300]}",
        f"**Previstas (top 15):** {', '.join(prev)}",
        f"**Acertos no top 15:** {acertos} de {len(real)} palavras reais",
        "",
    ]
texto = "\n".join(out)
print(texto, flush=True)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8").write(texto + "\n")
