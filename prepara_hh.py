"""
Gera pares_train_mix.jsonl a partir do hh-rlhf (harmless-base, treino).
Cada linha vira {"a": <última fala do Assistant no 'chosen'>}, que é o que o conta_mix.py lê.
Uso: ARATG_PASTA=. python prepara_hh.py hh/harmless_train.jsonl.gz [saida.jsonl]
"""
import gzip
import json
import os
import sys

PASTA = os.environ.get("ARATG_PASTA", ".")
SAIDA_PADRAO = os.path.join(PASTA, "pares_train_mix.jsonl")
MARCA = "\n\nAssistant:"


def resposta_final(texto):
    if MARCA not in texto:
        return None
    ultimo = texto.rsplit(MARCA, 1)[1]
    # corta a fala seguinte do humano, se existir
    ultimo = ultimo.split("\n\nHuman:")[0]
    ultimo = ultimo.strip()
    return ultimo or None


def main(entrada, saida=SAIDA_PADRAO):
    n = 0
    with gzip.open(entrada, "rt", encoding="utf-8") as f, \
         open(saida, "w", encoding="utf-8") as out:
        for linha in f:
            obj = json.loads(linha)
            a = resposta_final(obj["chosen"])
            if a is None:
                continue
            out.write(json.dumps({"a": a}, ensure_ascii=False) + "\n")
            n += 1
    print(f"pares gravados: {n} em {saida}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else SAIDA_PADRAO)
