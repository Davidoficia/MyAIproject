# Arquitetura do Nx (rascunho documentado, não fechado)

Status: **em aberto**. Este arquivo descreve o que o código faz hoje e o que foi medido.
Não é uma especificação final. Quando algo mudar no código, atualizar aqui.

## 1. O que é, em uma frase

Um modelo de **preenchimento de palavra central** por refinamento iterativo: dada uma janela
de 2 palavras à esquerda e 2 à direita, com o centro mascarado, ele produz uma distribuição
sobre o vocabulário e refina essa distribuição em K passes, realimentando o palpite macio
(média dos embeddings ponderada pela probabilidade) em cada passe.

Nome de trabalho no código: **Nx**. Termo "difusão" é usado no projeto por hábito, mas não
descreve bem o método: não há cronograma de ruído nem processo de difusão contínuo/discreto
no sentido da literatura. O mais honesto hoje é "refinamento iterativo de palavra mascarada"
(parecido com MaskGIT/mask-predict, sem as escolhas de cronograma deles).

## 2. Componentes (escala/nx_variantes.py)

| Parte | Forma | Função |
|---|---|---|
| Embedding `E` | V × 32 | um vetor por palavra; PAD fixo em zero |
| Contexto | 4 posições (2 esq + 2 dir) | `embed(E, ids)` de cada posição |
| Palpite macio | `P @ E` (B × 32) | embedding esperado do centro, refinado a cada passe |
| Entrada do MLP | concat de 5 blocos de 32 = 160 | 4 contextos + palpite |
| MLP | 160 → 128 (tanh) → V | `W1`, `W2`, com viés |
| Passe | `P_{k+1} = softmax(MLP(...))` | repetido K = 4 vezes, pesos compartilhados |
| Perda | entropia cruzada no centro | sobre `P_K` |

Constantes: D=32, HD=128, K=4; PAD=0, UNK=1, MASK=2.
Otimizador: Adam, taxa com aquecimento de 200 passos e cosseno até 10% da taxa máxima.

Não existem, no Nx base: atenção, LSTM, módulos, cabeça de parada. Eles existem em outras
variantes do mesmo arquivo (`mh`, `mod`, `lstm`), que não entraram na linha principal.

## 3. Vocabulário e entrada

- Palavras minúsculas por regex `\w+|[^\w\s]`, top 8000 do treino, fixas em `itos.json`.
- Palavras fora do vocabulário viram UNK. Taxa de UNK medida: ~12–14% no teste local,
  ~12,3% no quiz de 300 itens. Esse é o principal limite de cobertura hoje.
- Frases com menos de 3 tokens são descartadas.

## 4. Treino

- Amostra posições com alvo ≥ 3 (sem especiais); janela limitada à frase, fora da frase = PAD.
- Lote de 32, determinístico por (semente, passo), para retomada exata.
- Escala atual (CI): Wikipédia pt (TucanoBR, CC BY-SA 3.0) + 2.290 livros em domínio público.
- Retomada por `pesos/estado.npz`; `pesos/estado.txt` diz `continua` ou `fim`.
- Métricas durante o treino em `pesos/metricas.jsonl` (acerto como CI, bits/caractere por
  pseudo-log-verossimilhança, taxa de UNK, acerto por posição), a cada 50 mil passos.

## 5. Inferência / geração

- Preenchimento iterativo de posições com alvo. A cada passe, as posições mais confiantes
  são **fixadas** (estilo MaskGIT), com amostragem top-k e temperatura e penalidade de
  repetição. Sem fixar, a geração colapsava em vírgula e "de".
- Config usada nas medições: temperatura 0,9, top-k 30, penalidade 0,6 (conjunto de 15/11).
  Em outra medição, penalidade 1,5 zerou repetições seguidas sem perder diversidade.

## 6. O que foi medido (números do histórico, mesmo protocolo dentro de cada linha)

| Modelo | Dados | Acerto mascarado | Observação |
|---|---|---|---|
| Nx 3k passos | tema-separado | ~24% | comparativo de geração |
| Nx 954k (CI rodada 1) | Wikipédia + livros | 37,2% (teste do CI) | não comparável com os locais |
| Nx 1,43 mi (CI rodada 2) | idem | 38,4% (teste do CI) | +1,6 pt sobre 954k no diagnóstico local |
| Quiz de 30 itens, gabarito real | — | 954k: 16,7% | abaixo do acaso; n pequeno |

Ressalvas:
- Os números do CI e os locais usam testes e vocabulários diferentes. Só comparar dentro da mesma linha.
- Acerto com 1 passe cai a ~0%: o modelo depende dos 4 passes. Isso é um fato do desenho, não um defeito isolado.
- Coerência de frase: não medida de forma confiável. Os juízes automáticos (cabeças de tema e
  substituição) foram enviesados por frequência e por domínio. Ficaram fora da linha principal.
- Blind PT×EN e leitura cega: ver `/areas/aratg.md`.

## 7. Em aberto (ordem sugerida, sem compromisso)

1. Vocabulário: subpalavras como fallback (16k palavras + 2k subpalavras), para reduzir UNK.
   Comparar bits/caractere no mesmo texto e orçamento.
2. Janela: `escala/nx_janela.py` já tem W configurável e borda na fronteira da frase
   (default igual ao Nx atual, checado). Falta comparação no CI com orçamento igual.
3. Coerência entre passes: nenhuma solução medida ainda.
4. Quiz com gabarito real como critério de subir cabeças extras: continua sem passar.

## 8. Nome próprio (proposta, a escolher)

Sugestões, para o Davvzx decidir:
- **Garimpo**: a palavra certa sai de um conjunto de candidatos refinados em rodadas.
- **Lapidação**: cada passe tira imperfeições da palavra central (combina com o refinamento).
- **Eco-Lacuna**: lê o contexto dos dois lados e devolve a palavra que falta.

Enquanto não escolher, o nome de trabalho continua **Nx**.
