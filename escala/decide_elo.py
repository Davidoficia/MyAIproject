"""Decide se a cadeia do sentido continua. Uso: python3 decide_elo.py <historico.jsonl> <passos_totais> <max_total> <saida_estado.txt>
Para se: (a) a última avaliação não ganhou 0,3 ponto sobre a de duas avaliações atrás (platô), ou (b) bateu o total de passos."""
import json, sys
hist, total, maximo, saida = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
r = [json.loads(l)["real"] for l in open(hist, encoding="utf-8") if l.strip()]
motivo = "continua"
if total >= maximo:
    motivo = f"fim: passos {total} >= {maximo}"
elif len(r) >= 3 and (r[-1] - r[-3]) < 0.003:
    motivo = f"fim: platô (ganho em 2 avaliações = {(r[-1]-r[-3])*100:+.2f} pt)"
open(saida, "w", encoding="utf-8").write(("continua" if motivo == "continua" else "fim") + "\n")
print(motivo, "| avaliações:", [round(x * 100, 2) for x in r])
