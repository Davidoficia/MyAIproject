"""
Coleta N artigos aleatórios da Wikipédia pt (namespace 0, sem filtro de tema).
Guarda título, pageid, revid, URL e texto puro (extract), com atribuição CC BY-SA 3.0/4.0.
Split por ARTIGO: md5(título) % 20 == 0 vai pro teste (~5%).
Saída: OUT/artigos.json (artefato do CI, NÃO entra no repo).
Só biblioteca padrão.
"""
import hashlib, json, os, sys, time, urllib.parse, urllib.request

N = int(os.environ.get("N_ARTIGOS", "300"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "dados_aleatorios"
API = "https://pt.wikipedia.org/w/api.php"
UA = "aratg-dados/0.3 (https://github.com/Davidoficia/MyAIproject; projeto pessoal de pesquisa)"
os.makedirs(OUT, exist_ok=True)


def api(params):
    params = dict(params, format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for tent in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print("erro", tent, e, flush=True)
            time.sleep(3 * (tent + 1))
    raise RuntimeError("API falhou")


# 1) sorteio: pede aleatórios em lotes até juntar N páginas distintas
paginas = {}
while len(paginas) < N:
    r = api({"action": "query", "list": "random", "rnnamespace": 0,
             "rnlimit": min(50, N - len(paginas) + 10)})
    for p in r["query"]["random"]:
        paginas.setdefault(p["id"], p["title"])
    time.sleep(0.5)
ids = list(paginas)[:N]
print("sorteados:", len(ids), flush=True)

# 2) texto puro + revid + URL
artigos = []
# texto completo (explaintext) só retorna 1 extrato por requisição: uma página por chamada
for pid in ids:
    r = api({"action": "query", "pageids": str(pid), "prop": "extracts|revisions",
             "explaintext": "1", "rvprop": "ids", "redirects": "0"})
    for p in r["query"].get("pages", []):
        texto = (p.get("extract") or "").strip()
        if not texto:
            continue
        titulo = p["title"]
        artigos.append({
            "pageid": p["pageid"],
            "titulo": titulo,
            "revid": (p.get("revisions") or [{}])[0].get("revid"),
            "url": "https://pt.wikipedia.org/wiki/" + urllib.parse.quote(titulo.replace(" ", "_")),
            "licenca": "CC BY-SA 3.0 / 4.0 (Wikipédia pt); manter atribuição",
            "split": "teste" if int(hashlib.md5(titulo.encode()).hexdigest(), 16) % 20 == 0 else "treino",
            "texto": texto,
        })
    time.sleep(0.5)

with open(os.path.join(OUT, "artigos.json"), "w", encoding="utf-8") as f:
    json.dump(artigos, f, ensure_ascii=False, indent=1)
n_treino = sum(a["split"] == "treino" for a in artigos)
print(f"artigos com texto: {len(artigos)} (treino {n_treino}, teste {len(artigos) - n_treino})", flush=True)
print("chars:", sum(len(a["texto"]) for a in artigos), flush=True)
