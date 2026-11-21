"""
Inspeção de fontes (roda no CI). Amostra embaralhada (streaming com buffer) para não cair
no início ordenado dos datasets.
- Pt-Corpus: contagem de origens (campo metadata) numa amostra de 300k.
- VeraCruz: contagem de 'source' por configuração numa amostra de 200k cada.
Imprime contagens e proporções. Nenhum texto é impresso.
"""
import collections
import itertools
from datasets import load_dataset

SEMENTE = 0
BUFFER = 20_000

print("== Pt-Corpus: origens (amostra embaralhada, 300k) ==")
try:
    ds = load_dataset("nicholasKluge/Pt-Corpus", split="train", streaming=True)
    ds = ds.shuffle(seed=SEMENTE, buffer_size=BUFFER)
    cont = collections.Counter()
    total = 0
    for ex in itertools.islice(ds, 300_000):
        origem = str(ex.get("metadata", "")).replace("source: ", "")
        cont[origem[:100]] += 1
        total += 1
    for origem, n in cont.most_common(30):
        print(f"{n/total:7.2%}  {origem}")
    print(f"origens distintas: {len(cont)} | amostra: {total}")
except Exception as erro:
    print("ERRO Pt-Corpus:", repr(erro)[:300])

print("\n== VeraCruz_PT-BR: fonte por configuração (amostra embaralhada, 200k) ==")
for cfg in ["Portugal (PT)", "Brazil (BR)", "Other"]:
    try:
        ds = load_dataset("Bastao/VeraCruz_PT-BR", cfg, split="train", streaming=True)
        ds = ds.shuffle(seed=SEMENTE, buffer_size=BUFFER)
        cont = collections.Counter()
        total = 0
        for ex in itertools.islice(ds, 200_000):
            cont[str(ex.get("source", ""))] += 1
            total += 1
        print(f"-- {cfg} (amostra {total}):")
        for fonte, n in cont.most_common(15):
            print(f"   {n/total:7.2%}  {fonte}")
    except Exception as erro:
        print(f"{cfg}: ERRO {repr(erro)[:200]}")
