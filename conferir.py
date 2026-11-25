"""
Gera o "gabarito" do agente direto do CSV e do modelo do repositório (regressão OU classificação).
O professor roda este script no commit entregue de cada grupo para conferir os números da bateria.

    python conferir.py                  -> estatísticas + previsões das 3 primeiras linhas do dataset
    python conferir.py entradas.csv     -> previsões para as linhas de um CSV com as colunas de entrada
                                           (recomendado: use casos NOVOS, que não estão no dataset)

Funciona sem chave do Gemini. Mantenham ALVO, NUMERICAS e CATEGORICAS corretos em ferramentas.py.
"""
import sys

import pandas as pd

from ferramentas import ALVO, CATEGORICAS, DF, METRICAS, MODELO, NUMERICAS, conferir_transformacao, para_o_modelo

pd.set_option("display.width", 200, "display.max_columns", None)
CLASSIF = hasattr(MODELO, "classes_")          # classificador do sklearn (Pipeline incluso) tem classes_
alvo_num = pd.api.types.is_numeric_dtype(DF[ALVO])
print("== Métricas salvas:", METRICAS)
print("== Conferência da ponte legível -> modelo:", conferir_transformacao())
print(f"== Dataset: {DF.shape[0]} linhas x {DF.shape[1]} colunas | alvo: {ALVO} | tarefa: {'classificação' if CLASSIF else 'regressão'}")
print("\n== Estatísticas numéricas\n", DF[NUMERICAS + ([] if CLASSIF else [ALVO])].describe().round(2))
if CLASSIF:
    cont = DF[ALVO].value_counts()
    print(f"\n== Classes de {ALVO}\n", pd.DataFrame({"contagem": cont, "proporcao": (cont / len(DF)).round(4)}))
    print(f"\n== Médias das numéricas por classe de {ALVO}\n", DF.groupby(ALVO)[NUMERICAS].agg(["mean", "median"]).round(2).T)
    for col in CATEGORICAS:
        print(f"\n== Classes de {ALVO} por {col} (contagem)\n", pd.crosstab(DF[col], DF[ALVO], margins=True))
        print(f"\n== Classes de {ALVO} por {col} (proporção na linha)\n", pd.crosstab(DF[col], DF[ALVO], normalize="index").round(4))
else:
    for col in CATEGORICAS:
        print(f"\n== {ALVO} por {col}\n", DF.groupby(col)[ALVO].agg(["count", "mean", "median", "min", "max"]).round(2))
if alvo_num:
    print("\n== Correlação com o alvo (numéricas)\n", DF[NUMERICAS + [ALVO]].corr()[ALVO].drop(ALVO).round(3))
elif DF[ALVO].nunique() == 2:
    pos = sorted(DF[ALVO].unique())[-1]
    print(f"\n== Correlação das numéricas com {ALVO}=={pos!r} (0/1)\n", DF[NUMERICAS].corrwith((DF[ALVO] == pos).astype(int)).round(3))

entradas = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else DF.drop(columns=ALVO).head(3)
entradas = entradas[NUMERICAS + CATEGORICAS]
print("\n== Previsões do modelo")
saida = entradas.copy()
X_modelo = pd.concat([para_o_modelo(e) for e in entradas.to_dict("records")], ignore_index=True)   # PONTE
saida["previsao"] = MODELO.predict(X_modelo)
if hasattr(MODELO, "predict_proba"):
    probs = MODELO.predict_proba(X_modelo)
    for i, c in enumerate(MODELO.classes_):
        saida[f"prob_{c}"] = probs[:, i].round(3)
print(saida.to_string())
