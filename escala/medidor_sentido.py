"""Medidor de peças do motor de sentido: lê os resultados de cada braço e mede quanto cada peça ajudou.
Braços: TUDO (todas as peças) e SEM_<peça> (tudo menos aquela peça). Contribuição = TUDO - SEM_<peça>.
Uso: python3 medidor_sentido.py <pasta com res-*/resultado.json>"""
import glob, json, os, sys

pasta = sys.argv[1]
res = {}
for f in glob.glob(os.path.join(pasta, "**", "resultado.json"), recursive=True):
    r = json.load(open(f))
    res[r["arm"]] = r
metrica = "recall50_real"
linhas = ["| braço | pool | contexto | aquecimento | lote | recall@50 real | controle | alterada | mistura |", "|---|---|---|---|---|---|---|---|---|"]
for nome in ["BASE", "TUDO", "SEM_POOL", "SEM_OTIM", "SEM_CTX"]:
    if nome not in res:
        linhas.append(f"| {nome} | (não rodou) |  |  |  |  |  |  |  |")
        continue
    r = res[nome]
    linhas.append(f"| {nome} | {r['pool']} | {r['ctx']} | {r['warmup']} | {r['lote']} | {r[metrica]*100:.2f}% | {r['recall50_controle']*100:.2f}% | {r['recall50_alterada']*100:.2f}% | {r['recall50_mistura']*100:.2f}% |")
out = ["## Medidor de peças do sentido", ""] + linhas + [""]
if "TUDO" in res:
    t = res["TUDO"][metrica]
    out.append("### Quanto cada peça ajudou (TUDO − SEM_peça; positivo = a peça ajuda)")
    for peca, arm in [("atenção no resumo", "SEM_POOL"), ("lote grande + aquecimento", "SEM_OTIM"), ("contexto de 3 frases", "SEM_CTX")]:
        if arm in res:
            d = (t - res[arm][metrica]) * 100
            out.append(f"- **{peca}:** {d:+.2f} pontos")
    if "BASE" in res:
        out.append(f"- **total (TUDO − BASE):** {(t - res['BASE'][metrica]) * 100:+.2f} pontos")
    out.append("")
    out.append("Ressalva: uma semente só. Diferenças menores que ~0,5 ponto podem ser ruído.")
texto = "\n".join(out)
print(texto)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8").write(texto + "\n")
