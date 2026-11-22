"""
Acrescenta livros em domínio público (PleIAs/Portuguese-PD, declarado no README como domínio
público em todas as regiões; obras de autores mortos há +70 anos e publicadas até 1884) a
dados/treino.txt e teste.txt, uma frase por linha.
Split por LIVRO: md5(identificador) % 20 == 0 vai para o teste.
Limite de palavras (MAX_LIVROS_PALAVRAS) para caber na RAM do runner.
Uso: python3 -I escala/prepara_livros.py <pasta_dados>
"""
import hashlib, json, os, re, sys, urllib.request
import pyarrow.parquet as pq

REPO = "PleIAs/Portuguese-PD"
OUT = sys.argv[1]
MAX_PAL = int(os.environ.get("MAX_LIVROS_PALAVRAS", "150000000"))
UA = "aratg-dados/0.3 (https://github.com/Davidoficia/myaiproject; projeto pessoal de pesquisa)"
FRASE = re.compile(r"(?<=[.!?])\s+")


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


req = urllib.request.Request(f"https://huggingface.co/api/datasets/{REPO}", headers={"User-Agent": UA})
info = json.load(urllib.request.urlopen(req, timeout=120))
arquivos = sorted(s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".parquet"))
print("parquets de livros:", len(arquivos), flush=True)

tr = open(os.path.join(OUT, "treino.txt"), "a", encoding="utf-8")
te = open(os.path.join(OUT, "teste.txt"), "a", encoding="utf-8")
palavras, n_liv_tr, n_liv_te, n_fr = 0, 0, 0, 0
for arq in arquivos:
    if palavras >= MAX_PAL:
        break
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{arq}"
    local = os.path.join(OUT, os.path.basename(arq))
    print("baixando", arq, flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=900) as r, open(local, "wb") as out:
        while True:
            pedaco = r.read(1 << 20)
            if not pedaco:
                break
            out.write(pedaco)
    try:
        pf = pq.ParquetFile(local)
        colunas = [c for c in ["identifier", "text"] if c in pf.schema_arrow.names]
        if "text" not in colunas:
            print("sem coluna text, pulando", arq, flush=True); os.remove(local); continue
        for lote in pf.iter_batches(batch_size=500, columns=colunas):
            d = lote.to_pydict()
            ids = d.get("identifier") or [str(i) for i in range(len(d["text"]))]
            for ident, txt in zip(ids, d["text"]):
                if not txt or palavras >= MAX_PAL:
                    continue
                fr = frases(txt)
                if not fr:
                    continue
                teste_livro = int(hashlib.md5(str(ident).encode()).hexdigest(), 16) % 20 == 0
                dest = te if teste_livro else tr
                for f in fr:
                    dest.write(f + "\n")
                    palavras += len(f.split()); n_fr += 1
                if teste_livro: n_liv_te += 1
                else: n_liv_tr += 1
    except Exception as erro:
        print("ERRO em", arq, repr(erro)[:200], flush=True)
    os.remove(local)
tr.close(); te.close()
print(f"livros treino: {n_liv_tr} | livros teste: {n_liv_te} | frases: {n_fr} | palavras: {palavras}", flush=True)
