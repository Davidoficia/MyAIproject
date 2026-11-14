"""
Avalia o modelo de contagem no TESTE (split por pergunta), palavra exata, previsão
com contexto real (teacher forcing). Modelos:
  uni : palavra mais frequente do treino
  big : bigrama (1 palavra), recua pro unigrama
  tri : trigrama (2 palavras), recua pro bigrama, depois unigrama
Quebra por fonte (instrucao / mensagens).
"""
import json
import os
import re
import sqlite3
from collections import defaultdict

PASTA = os.environ.get("ARATG_PASTA", os.path.expanduser("~/aratg_pipe"))
TESTE = os.environ.get("ARATG_TESTE", os.path.join(PASTA, "pares_test_limpo2.jsonl"))
TOKEN = re.compile(r"\w+|[^\w\s]")


def tokens(texto):
    return TOKEN.findall(texto.lower())


con = sqlite3.connect(os.path.join(PASTA, "contagens_mix.sqlite"))
con.execute("PRAGMA query_only=ON")
UNI_TOP = con.execute("SELECT t FROM uni ORDER BY n DESC LIMIT 1").fetchone()[0]
cache_bi, cache_tri = {}, {}


def top_bi(p1):
    if p1 not in cache_bi:
        r = con.execute("SELECT t FROM bi WHERE p=? ORDER BY n DESC LIMIT 1", (p1,)).fetchone()
        cache_bi[p1] = r[0] if r else None
    return cache_bi[p1]


def top_tri(p2, p1):
    k = (p2, p1)
    if k not in cache_tri:
        r = con.execute("SELECT t FROM tri WHERE p2=? AND p1=? ORDER BY n DESC LIMIT 1", k).fetchone()
        cache_tri[k] = r[0] if r else None
    return cache_tri[k]


def preve(modelo, p2, p1):
    if modelo == "uni":
        return UNI_TOP
    if modelo == "big":
        return top_bi(p1) or UNI_TOP
    t = top_tri(p2, p1)
    return t or top_bi(p1) or UNI_TOP


total = defaultdict(int)
acerto = defaultdict(lambda: defaultdict(int))
modelos = ["uni", "big", "tri"]
with open(TESTE, "r", encoding="utf-8") as f:
    for linha in f:
        obj = json.loads(linha)
        fonte = obj.get("fonte", "geral")
        toks = ["<INI>"] + tokens(obj["a"]) + ["<FIM>"]
        for j in range(1, len(toks)):
            p1 = toks[j - 1]
            p2 = toks[j - 2] if j >= 2 else "<INI>"
            alvo = toks[j]
            for m in modelos:
                ok = int(preve(m, p2, p1) == alvo)
                acerto[m][fonte] += ok
                acerto[m]["todos"] += ok
            total[fonte] += 1
            total["todos"] += 1
print("posições de teste:", {k: v for k, v in total.items()}, flush=True)
for m in modelos:
    for fonte in ("instrucao", "mensagens", "geral", "todos"):
        n = total[fonte]
        if n:
            print(f"{m:4s} {fonte:10s} acerto palavra exata: {acerto[m][fonte] / n:.1%}", flush=True)
