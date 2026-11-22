"""Prepara inglês para o teste PT x EN: public_library_1929_dolma (domínio público nos EUA, pré-1929;
o README admite metadados imprecisos de licença). Usa só alguns arquivos (o total tem ~74 GB).
Split por documento: md5(id) % 20 == 0 -> teste. Uso: python3 -I escala/prepara_en.py <pasta_saida>"""
import gzip, hashlib, json, os, re, sys, urllib.request
import pyarrow.parquet as pq
REPO = "storytracer/public_library_1929_dolma"
OUT = sys.argv[1]
MAX_ARQ = int(os.environ.get("MAX_ARQUIVOS", "2"))
UA = "aratg-dados/0.3 (teste PT x EN; projeto pessoal de pesquisa)"
FRASE = re.compile(r"(?<=[.!?])\s+")
os.makedirs(OUT, exist_ok=True)


def frases(texto):
    saida = []
    for linha in texto.split("\n"):
        for f in FRASE.split(linha.strip()):
            p = f.split()
            if len(p) < 6 or len(p) > 60 or "http" in f:
                continue
            if sum(c.isalpha() for c in f) / max(1, len(f)) < 0.6:
                continue
            saida.append(f.strip())
    return saida


info = json.load(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{REPO}", headers={"User-Agent": UA}), timeout=120))
arqs = sorted(s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith((".parquet", ".jsonl.gz", ".jsonl")))
print("arquivos encontrados:", len(arqs), "| usando", min(MAX_ARQ, len(arqs)), flush=True)
tr = open(os.path.join(OUT, "treino.txt"), "w", encoding="utf-8")
te = open(os.path.join(OUT, "teste.txt"), "w", encoding="utf-8")
n_doc = 0
for arq in arqs[:MAX_ARQ]:
    local = os.path.join(OUT, "_" + os.path.basename(arq))
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{arq}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=900) as r, open(local, "wb") as out:
        while True:
            b = r.read(1 << 20)
            if not b: break
            out.write(b)
    docs = []
    if arq.endswith(".parquet"):
        t = pq.read_table(local, columns=[c for c in ["id", "text"] if c in pq.read_schema(local).names]).to_pydict()
        docs = list(zip(t.get("id", [str(i) for i in range(len(t["text"]))]), t["text"]))
    else:
        op = gzip.open if arq.endswith(".gz") else open
        with op(local, "rt", encoding="utf-8") as f:
            for i, l in enumerate(f):
                d = json.loads(l); docs.append((d.get("id", f"{arq}:{i}"), d.get("text", "")))
    for ident, txt in docs:
        if not txt: continue
        fr = frases(txt)
        if not fr: continue
        dest = te if int(hashlib.md5(str(ident).encode()).hexdigest(), 16) % 20 == 0 else tr
        for f in fr: dest.write(f + "\n")
        n_doc += 1
    os.remove(local)
tr.close(); te.close()
print(f"documentos usados: {n_doc}", flush=True)
