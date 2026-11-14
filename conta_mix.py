"""
Contagem de unigramas, bigramas e trigramas SOBRE O TREINO, em streaming.
- Acumula em memória por lote (LOTE linhas) e grava em bloco no SQLite (rápido).
- Retomável: o checkpoint só avança junto com um commit. Se cair, roda de novo e
  continua da última linha commitada (trabalho do lote em curso é descartado e refeito).
"""
import json
import os
import re
import sqlite3
from collections import Counter

PASTA = os.environ.get("ARATG_PASTA", os.path.expanduser("~/aratg_pipe"))
BANCO = os.path.join(PASTA, "contagens_mix.sqlite")
ENTRADA = os.path.join(PASTA, "pares_train_mix.jsonl")
LOTE = 1000
TOKEN = re.compile(r"\w+|[^\w\s]")


def tokens(texto):
    return TOKEN.findall(texto.lower())


def abre():
    con = sqlite3.connect(BANCO)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE IF NOT EXISTS uni(t TEXT PRIMARY KEY, n INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS bi(p TEXT, t TEXT, n INTEGER, PRIMARY KEY(p,t))")
    con.execute("CREATE TABLE IF NOT EXISTS tri(p2 TEXT, p1 TEXT, t TEXT, n INTEGER, PRIMARY KEY(p2,p1,t))")
    con.execute("CREATE TABLE IF NOT EXISTS ck(id INTEGER PRIMARY KEY CHECK(id=1), linha INTEGER)")
    con.execute("INSERT OR IGNORE INTO ck(id, linha) VALUES (1, 0)")
    con.commit()
    return con


def grava_lote(con, uni, bi, tri, linha_final):
    con.executemany("INSERT INTO uni(t,n) VALUES(?,?) ON CONFLICT(t) DO UPDATE SET n=n+excluded.n",
                    uni.items())
    con.executemany("INSERT INTO bi(p,t,n) VALUES(?,?,?) ON CONFLICT(p,t) DO UPDATE SET n=n+excluded.n",
                    [(p, t, n) for (p, t), n in bi.items()])
    con.executemany("INSERT INTO tri(p2,p1,t,n) VALUES(?,?,?,?) "
                    "ON CONFLICT(p2,p1,t) DO UPDATE SET n=n+excluded.n",
                    [(a, b, c, n) for (a, b, c), n in tri.items()])
    con.execute("UPDATE ck SET linha=? WHERE id=1", (linha_final,))
    con.commit()


def main():
    con = abre()
    feitas = con.execute("SELECT linha FROM ck WHERE id=1").fetchone()[0]
    print(f"retomando na linha {feitas}", flush=True)
    uni, bi, tri = Counter(), Counter(), Counter()
    no_lote = 0
    ultimo = feitas
    with open(ENTRADA, "r", encoding="utf-8") as f:
        for i, linha in enumerate(f):
            if i < feitas:
                continue
            obj = json.loads(linha)
            toks = ["<INI>"] + tokens(obj["a"]) + ["<FIM>"]
            for t in toks[1:]:
                uni[t] += 1
            for j in range(1, len(toks)):
                bi[(toks[j - 1], toks[j])] += 1
                if j >= 2:
                    tri[(toks[j - 2], toks[j - 1], toks[j])] += 1
            no_lote += 1
            ultimo = i + 1
            if no_lote == LOTE:
                grava_lote(con, uni, bi, tri, ultimo)
                uni, bi, tri, no_lote = Counter(), Counter(), Counter(), 0
                print(f"  linha {ultimo}", flush=True)
    if no_lote:
        grava_lote(con, uni, bi, tri, ultimo)
    total = con.execute("SELECT COUNT(*) FROM bi").fetchone()[0]
    print(f"concluído na linha {ultimo} | pares bigrama distintos: {total}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
