"""
Inspeciona o esquema dos candidatos a base nova (roda no CI, onde o HF responde).
Imprime só nomes de colunas e um trecho curto de 2 exemplos por fonte.
Objetivo: achar o campo de fonte/licença para filtrar Wikipédia e NC/ND.
"""
import itertools
from datasets import load_dataset

CANDIDATOS = [
    ("nicholasKluge/Pt-Corpus", None, "train"),
    ("Bastao/VeraCruz_PT-BR", None, "train"),
    ("oscar-corpus/OSCAR-2301", "pt", "train"),
    ("uoft-cs/cultura-x", "pt", "train"),
]

for nome, config, split in CANDIDATOS:
    try:
        ds = load_dataset(nome, config, split=split, streaming=True, trust_remote_code=False)
        ex = list(itertools.islice(ds, 2))
        print(f"\n== {nome} ({config}) ==")
        print("colunas:", list(ex[0].keys()))
        for e in ex:
            print({k: str(v)[:80] for k, v in e.items()})
    except Exception as erro:
        print(f"\n== {nome} ({config}) == ERRO: {repr(erro)[:300]}")
