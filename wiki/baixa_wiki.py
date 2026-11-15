"""
Baixa artigos da Wikipédia pt pela API (texto puro, explaintext=1), limpa e grava
um arquivo por tema, uma frase por linha.

Só biblioteca padrão. Uso:
    python3 wiki/baixa_wiki.py --saida wiki_dados
"""
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

API = "https://pt.wikipedia.org/w/api.php"
UA = "aratg-dados/0.1 (https://github.com/Davidoficia/MyAIproject; projeto pessoal de pesquisa)"

TEMAS = {
    "animais": [
        "Tucano", "Onça-pintada", "Capivara", "Arara-azul", "Sapo", "Borboleta", "Formiga",
        "Abelha", "Baleia-jubarte", "Tartaruga-marinha", "Lobo-guará", "Preguiça-de-coleira",
        "Tamanduá-bandeira", "Beija-flor", "Pinguim", "Golfinho", "Coruja", "Polvo", "Cavalo",
        "Gato", "Cão", "Vaca", "Galinha", "Elefante", "Girafa", "Leão", "Morcego", "Libélula",
        "Jacaré", "Pirarucu",
    ],
    "cidades": [
        "São Paulo", "Rio de Janeiro", "Salvador", "Belo Horizonte", "Curitiba", "Manaus",
        "Recife", "Porto Alegre", "Fortaleza", "Brasília", "Belém", "Goiânia",
        "Florianópolis", "Natal", "João Pessoa", "Maceió", "Vitória", "Campo Grande",
        "Cuiabá", "Porto Velho", "Macapá", "Boa Vista", "Palmas", "Teresina", "São Luís",
        "Aracaju", "Rio Branco", "Ouro Preto", "Paraty", "Foz do Iguaçu",
    ],
    "historia": [
        "Descobrimento do Brasil", "Brasil Colônia", "Império do Brasil",
        "Proclamação da República no Brasil", "Guerra do Paraguai",
        "Abolição da escravidão no Brasil", "Era Vargas", "Ditadura militar no Brasil",
        "Constituição brasileira de 1988", "Inconfidência Mineira", "Revolta da Vacina",
        "Independência do Brasil", "Bandeirantes", "Quilombo dos Palmares",
        "Revolução Farroupilha", "Estado Novo", "Diretas Já", "Plano Real",
        "Guerra de Canudos", "Semana de Arte Moderna",
    ],
    "ciencia": [
        "Fotossíntese", "Gravidade", "Átomo", "Célula", "Energia", "Eletricidade", "Calor",
        "Luz", "Evolução biológica", "Sistema Solar", "Lua", "Terra", "Água", "Oxigênio",
        "Clorofila", "DNA", "Ecossistema", "Tabela periódica", "Temperatura", "Magnetismo",
        "Termodinâmica", "Fóton", "Osmose", "Vulcão", "Terremoto", "Clima", "Atmosfera",
        "Nuvem", "Chuva", "Vento",
    ],
}

# seções que não são prosa: corta o resto do artigo a partir delas
CORTE = {
    "ver também", "referências", "referências bibliográficas", "bibliografia",
    "ligações externas", "notas", "notas e referências", "leitura complementar",
    "referências e notas", "fontes",
}
HEADING = re.compile(r"^==+\s*(.*?)\s*==+\s*$")
CITACAO = re.compile(r"\[\s*\d+\s*\]|\[nota \d+\]|\[\s*[a-z]\s*\]")
FRASE = re.compile(r"(?<=[.!?])\s+")


def pede_lote(titulos):
    """Pede um lote de títulos. Trata 429 esperando mais; não engole erro."""
    params = {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "exlimit": "max", "redirects": "1", "format": "json", "formatversion": "2",
        "titles": "|".join(titulos),
    }
    url = API + "?" + urllib.parse.urlencode(params)
    espera = 30
    for tentativa in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                print(f"  {e.code}, esperando {espera}s", flush=True)
                time.sleep(espera)
                espera = min(espera * 2, 300)
                continue
            raise
        except Exception as e:
            print(f"  erro ({tentativa+1}/6): {type(e).__name__} {e}", flush=True)
            time.sleep(10)
    raise RuntimeError("API falhou depois de 6 tentativas: " + "|".join(titulos)[:120])


def limpa(texto):
    frases_saida = []
    for linha in texto.split("\n"):
        linha = linha.strip()
        if not linha:
            continue
        m = HEADING.match(linha)
        if m:
            if m.group(1).strip().lower() in CORTE:
                break  # resto do artigo é bibliografia/links
            continue  # título de seção não é prosa
        linha = CITACAO.sub("", linha)
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
            frases_saida.append(f)
    return frases_saida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="wiki_dados")
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)
    stats = {}
    for tema, titulos in TEMAS.items():
        vistos, frases, achados, faltando = set(), [], 0, []
        for i in range(0, len(titulos), 10):
            lote = titulos[i:i + 10]
            dados = pede_lote(lote)
            paginas = dados.get("query", {}).get("pages", [])
            if i == 0:
                print("  DEBUG chaves:", list(dados.keys()), json.dumps(dados)[:500], flush=True)
            achados_lote = set()
            for pg in paginas:
                if pg.get("missing") or not pg.get("extract"):
                    continue
                achados += 1
                achados_lote.add(pg.get("title"))
                for f in limpa(pg["extract"]):
                    if f not in vistos:
                        vistos.add(f)
                        frases.append(f)
            for t in lote:
                if not any(t.lower() == a.lower() or t.lower() in a.lower() for a in achados_lote):
                    faltando.append(t)
            time.sleep(3)
        caminho = os.path.join(args.saida, f"wiki_{tema}.txt")
        with open(caminho, "w", encoding="utf-8") as out:
            out.write("\n".join(frases) + "\n")
        stats[tema] = {"pedidos": len(titulos), "achados": achados,
                       "faltando": faltando, "frases": len(frases),
                       "palavras": sum(len(f.split()) for f in frases)}
        print(f"{tema}: {achados}/{len(titulos)} artigos, {len(frases)} frases", flush=True)
    with open(os.path.join(args.saida, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
