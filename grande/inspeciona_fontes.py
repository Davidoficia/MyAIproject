"""
Inspeciona o esquema e as fontes dos candidatos a base nova (roda no CI, onde o HF responde).
- Pt-Corpus: conta as URLs de origem (campo metadata) numa amostra, para filtrar Wikipédia e NC/ND.
- VeraCruz: lista as configurações (Portugal, Brasil, Other) e mostra as colunas de cada uma.
- OSCAR e CulturaX: tenta carregar e mostra o erro (gated ou inexistente).
Imprime só nomes de colunas, contagens e trechos curtos.
"""
import collections
import itertools
from datasets import load_dataset

print("== Pt-Corpus: fontes na amostra de 300k documentos ==")
try:
    ds = load_dataset("nicholasKluge/Pt-Corpus", split="train", streaming=True)
    cont = collections.Counter()
    for ex in itertools.islice(ds, 300_000):
        meta = str(ex.get("metadata", ""))
        cont[meta.replace("source: ", "")[:90]] += 1
    for fonte, n in cont.most_common(25):
        print(f"{n:7d}  {fonte}")
    print(f"fontes distintas na amostra: {len(cont)}")
except Exception as erro:
    print("ERRO Pt-Corpus:", repr(erro)[:300])

print("\n== VeraCruz_PT-BR: configurações ==")
for cfg in ["Portugal (PT)", "Brazil (BR)", "Other"]:
    try:
        ds = load_dataset("Bastao/VeraCruz_PT-BR", cfg, split="train", streaming=True)
        ex = next(iter(ds))
        print(f"{cfg}: colunas {list(ex.keys())} | amostra {{ {', '.join(f'{k}: {str(v)[:60]!r}' for k, v in ex.items())} }}")
    except Exception as erro:
        print(f"{cfg}: ERRO {repr(erro)[:200]}")

print("\n== OSCAR e CulturaX ==")
for nome, cfg in [("oscar-corpus/OSCAR-2301", "pt"), ("uoft-cs/cultura-x", "pt")]:
    try:
        ds = load_dataset(nome, cfg, split="train", streaming=True)
        print(nome, "OK, colunas:", list(next(iter(ds)).keys()))
    except Exception as erro:
        print(f"{nome}: ERRO {repr(erro)[:200]}")
