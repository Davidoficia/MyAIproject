"""
Mede a taxa de <UNK> com o mesmo vocabulário do Nx (top 8000 do 1M de frases de treino).
Mede no teste inteiro e num amostra do treino, e lista as palavras fora do vocabulário mais frequentes.
Variáveis de ambiente: DADOS.
"""
import json, os, re
from collections import Counter

DIR = os.environ.get("DADOS", "dados_grande")
L = 32; VOC = 8000
TOK = re.compile(r"\w+|[^\w\s]")
tokens = lambda s: TOK.findall(s.lower())[:L]

cont = Counter()
with open(f"{DIR}/treino.txt", encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i >= 1_000_000: break
        cont.update(tokens(l))
vocab = {w for w, _ in cont.most_common(VOC)}
print(f"vocabulário: {len(vocab)} palavras", flush=True)

def mede(caminho, limite=None):
    tot = unk = frases = 0
    fora = Counter()
    with open(caminho, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if limite is not None and i >= limite: break
            toks = tokens(l)
            if len(toks) < 3: continue
            frases += 1
            for w in toks:
                tot += 1
                if w not in vocab:
                    unk += 1; fora[w] += 1
    return {"frases": frases, "tokens": tot, "unk": unk, "taxa": round(unk / max(1, tot), 4),
            "mais_fora": fora.most_common(15)}

resultado = {"treino_1M_frases": mede(f"{DIR}/treino.txt", 1_000_000),
             "teste_inteiro": mede(f"{DIR}/teste.txt")}
print("RESUMO", json.dumps(resultado, ensure_ascii=False), flush=True)
print("FIM", flush=True)
