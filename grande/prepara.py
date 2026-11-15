"""
Baixa os parquets da Wikipédia pt (TucanoBR/wikipedia-PT, CC BY-SA 3.0) do Hugging Face,
limpa e grava uma frase por linha em treino.txt e teste.txt.
Split por ARTIGO: ~5% dos artigos (md5 do título % 20 == 0) vão pro teste.
Só biblioteca padrão + pyarrow.
"""
import hashlib, json, os, re, sys, urllib.request
import pyarrow.parquet as pq

REPO = "TucanoBR/wikipedia-PT"
OUT = sys.argv[1] if len(sys.argv) > 1 else "dados_grande"
os.makedirs(OUT, exist_ok=True)
FRASE = re.compile(r"(?<=[.!?])\s+")
CORTE = {"referências", "referencias", "ver também", "bibliografia", "ligações externas",
         "notas", "notas e referências", "leitura complementar", "fontes"}
UA = "aratg-dados/0.2 (https://github.com/Davidoficia/MyAIproject; projeto pessoal de pesquisa)"


def limpa(texto):
    saida = []
    for linha in texto.split("\n"):
        linha = linha.strip()
        if not linha:
            continue
        if linha.lower().strip("= ") in CORTE:
            break
        linha = re.sub(r"\[\s*\d+\s*\]", "", linha)
        for f in FRASE.split(linha):
            f = f.strip()
            palavras = f.split()
            if len(palavras) < 6 or len(palavras) > 60:
                continue
            if "http" in f or "|" in f or "{" in f:
                continue
            letras = sum(c.isalpha() for c in f)
            if letras / max(1, len(f)) < 0.6:
                continue
            saida.append(f)
    return saida


req = urllib.request.Request(f"https://huggingface.co/api/datasets/{REPO}", headers={"User-Agent": UA})
info = json.load(urllib.request.urlopen(req, timeout=120))
arquivos = [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".parquet")]
print("parquets:", len(arquivos), flush=True)

treino = open(os.path.join(OUT, "treino.txt"), "w", encoding="utf-8")
teste = open(os.path.join(OUT, "teste.txt"), "w", encoding="utf-8")
n_art = n_fr_tr = n_fr_te = 0
for arq in arquivos:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{arq}"
    local = os.path.join(OUT, os.path.basename(arq))
    print("baixando", arq, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r, open(local, "wb") as out:
        while True:
            pedaco = r.read(1 << 20)
            if not pedaco:
                break
            out.write(pedaco)
    pf = pq.ParquetFile(local)
    coluna_txt = None
    for lote in pf.iter_batches(batch_size=2000):
        d = lote.to_pydict()
        if coluna_txt is None:
            coluna_txt = "text" if "text" in d else [k for k, v in d.items() if v and isinstance(v[0], str)][0]
            print("coluna de texto:", coluna_txt, "| colunas:", list(d.keys()), flush=True)
        titulos = d.get("title") or [str(i) for i in range(len(d[coluna_txt]))]
        for txt, titulo in zip(d[coluna_txt], titulos):
            if not txt:
                continue
            frases = limpa(txt)
            if not frases:
                continue
            n_art += 1
            dest = teste if int(hashlib.md5(str(titulo).encode()).hexdigest(), 16) % 20 == 0 else treino
            for f in frases:
                dest.write(f + "\n")
            if dest is teste: n_fr_te += len(frases)
            else: n_fr_tr += len(frases)
    os.remove(local)
treino.close(); teste.close()
print(f"artigos úteis: {n_art} | frases treino: {n_fr_tr} | frases teste: {n_fr_te}", flush=True)
