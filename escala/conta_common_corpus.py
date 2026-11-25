"""Conta documentos em português com domínio público no PleIAs/common_corpus (amostra em streaming).
Uso: N=200000 python3 conta_common_corpus.py <saida.json>
Imprime o esquema real do dataset antes de contar (os nomes dos campos são descobertos, não chutados)."""
import collections, json, os, sys
from datasets import load_dataset

saida = sys.argv[1]
N = int(os.environ.get("N", "200000"))
ds = load_dataset("PleIAs/common_corpus", split="train", streaming=True)

lang_c, lic_c, col_c = collections.Counter(), collections.Counter(), collections.Counter()
pt_total = pt_pd = 0
pt_pd_palavras = 0
primeiro = None
for i, row in enumerate(ds):
    if i >= N:
        break
    if primeiro is None:
        primeiro = {k: (str(v)[:80] if not isinstance(v, (int, float)) else v) for k, v in row.items()}
        print("ESQUEMA:", json.dumps(primeiro, ensure_ascii=False, indent=1), flush=True)
    lang = str(row.get("language", "")).lower()
    lic = str(row.get("license", "")).lower()
    col = str(row.get("collection", ""))
    lang_c[lang] += 1
    lic_c[lic] += 1
    col_c[col] += 1
    if "portug" in lang or lang in ("pt", "por"):
        pt_total += 1
        if "public" in lic or "domain" in lic or lic in ("pd", "cc0"):
            pt_pd += 1
            pt_pd_palavras += len(str(row.get("text", "")).split())

res = {
    "amostra_docs": min(N, i + 1),
    "pt_total": pt_total,
    "pt_dominio_publico": pt_pd,
    "pt_dominio_publico_palavras": pt_pd_palavras,
    "top_idiomas": lang_c.most_common(15),
    "top_licencas": lic_c.most_common(15),
    "top_colecoes": col_c.most_common(10),
    "esquema": primeiro,
}
json.dump(res, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ["amostra_docs", "pt_total", "pt_dominio_publico", "pt_dominio_publico_palavras"]}, indent=1), flush=True)
print("TOP LICENCAS:", lic_c.most_common(15), flush=True)
