"""
Converte um parquet do Tulu (coluna 'messages') em pares {"a": <última fala do assistente>}.
Uso: ARATG_PASTA=. python prepara_tulu.py dados.parquet [saida.jsonl]
"""
import json
import os
import sys

import pyarrow.parquet as pq

PASTA = os.environ.get("ARATG_PASTA", ".")


def ultima_resposta(msgs):
    for m in reversed(msgs):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m["content"].strip()
    return None


def main(entrada, saida=None):
    saida = saida or os.path.join(PASTA, "pares_train_mix.jsonl")
    tab = pq.read_table(entrada, columns=["messages"]).to_pylist()
    n = 0
    with open(saida, "w", encoding="utf-8") as out:
        for linha in tab:
            a = ultima_resposta(linha["messages"] or [])
            if a:
                out.write(json.dumps({"a": a}, ensure_ascii=False) + "\n")
                n += 1
    print(f"pares gravados: {n} em {saida}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
