"""
Treino do modelo do Trabalho Final (mesmo dataset do Trabalho Prático: Default of Credit Card Clients).

    python treinar.py

O que este script faz (tudo com random_state=42, para ser reproduzível):
  1. lê dados/default_of_credit_card_clients.csv e faz a MESMA limpeza do notebook do Trabalho Prático
     (remove duplicatas exatas, corrige categorias fora do dicionário, renomeia colunas);
  2. grava dados/dataset_legivel.csv (texto por extenso e escala original: é o que o agente e o RAG leem);
  3. treina Regressão Logística, KNN e MLP com Pipeline (StandardScaler + OneHotEncoder dentro do Pipeline)
     e GridSearchCV (5 dobras, F1), usando só 8 atributos de entrada (limite do enunciado);
  4. escolhe o melhor modelo pelo F1 da VALIDAÇÃO CRUZADA (no treino), e só depois mede no teste;
  5. salva modelo/modelo.joblib (Pipeline completo), modelo/metricas.json, modelo/amostra_conferencia.json
     (para conferir a ponte para_o_modelo) e conhecimento/modelo_e_metricas.md (texto gerado para o RAG).
"""
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*penalty.*")      # aviso de depreciação do liblinear no sklearn 1.9 (funciona)
os.chdir(Path(__file__).resolve().parent)
SEED = 42

# ------------------------------------------------------------------ 1. limpeza (igual ao Trabalho Prático)
df = pd.read_csv("dados/default_of_credit_card_clients.csv", header=1)
df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "default"}).drop(columns="ID")
n0 = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicatas removidas: {n0 - len(df)} -> {len(df)} linhas")
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})      # valores fora do dicionário viram "outros"
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# ------------------------------------------------------------------ 2. dataset legível (para pessoas e RAG)
legivel = pd.DataFrame({
    "limite_credito": df["LIMIT_BAL"],                  # NT$ (novo dólar taiwanês)
    "idade": df["AGE"],
    "escolaridade": df["EDUCATION"].map({1: "Pós-graduação", 2: "Universidade", 3: "Ensino médio", 4: "Outros"}),
    "situacao_pag_set": df["PAY_1"],                    # situação do pagamento em set/2005 (-2 a 9)
    "situacao_pag_ago": df["PAY_2"],
    "situacao_pag_jul": df["PAY_3"],
    "fatura_set": df["BILL_AMT1"],                      # valor da fatura de set/2005 (NT$)
    "pagamento_set": df["PAY_AMT1"],                    # valor pago em set/2005 (NT$)
    "sexo": df["SEX"].map({1: "Masculino", 2: "Feminino"}),      # só para estatísticas (NÃO é entrada do modelo)
    "estado_civil": df["MARRIAGE"].map({1: "Casado", 2: "Solteiro", 3: "Outros"}),   # idem
    "inadimplente": np.where(df["default"] == 1, "sim", "nao"),
})
legivel.to_csv("dados/dataset_legivel.csv", index=False)

# ------------------------------------------------------------------ 3. treino
NUMERICAS = ["limite_credito", "idade", "situacao_pag_set", "situacao_pag_ago", "situacao_pag_jul",
             "fatura_set", "pagamento_set"]
CATEGORICAS = ["escolaridade"]
ALVO = "inadimplente"
X, y = legivel[NUMERICAS + CATEGORICAS], legivel[ALVO]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
print("Treino:", X_tr.shape, "| Teste:", X_te.shape, f"| inadimplentes no treino: {(y_tr == 'sim').mean() * 100:.2f}%")


def pipe(clf):
    prep = ColumnTransformer([("num", StandardScaler(), NUMERICAS),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAS)])
    return Pipeline([("prep", prep), ("clf", clf)])      # o scaler só aprende com o treino (sem data leakage)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
F1_SIM = make_scorer(f1_score, pos_label="sim")           # o alvo é texto ("sim"/"nao"): precisa dizer qual é a classe positiva
MODELOS = {
    "Regressão Logística": (pipe(LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED)),
                            {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1", "l2"]}),
    "KNN": (pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 11, 21, 31, 51], "clf__weights": ["uniform", "distance"],
             "clf__metric": ["euclidean", "manhattan"]}),
    "MLP": (pipe(MLPClassifier(max_iter=500, early_stopping=True, n_iter_no_change=15, random_state=SEED)),
            {"clf__hidden_layer_sizes": [(32,), (64,), (64, 32)], "clf__activation": ["relu", "tanh"],
             "clf__alpha": [1e-4, 1e-3, 1e-2]}),
}


def metricas_teste(modelo) -> dict:
    prev = modelo.predict(X_te)
    prob = modelo.predict_proba(X_te)[:, list(modelo.classes_).index("sim")]
    vn, fp, fn, vp = confusion_matrix(y_te, prev, labels=["nao", "sim"]).ravel()
    return {"acuracia": accuracy_score(y_te, prev), "precisao": precision_score(y_te, prev, pos_label="sim"),
            "recall": recall_score(y_te, prev, pos_label="sim"), "especificidade": vn / (vn + fp),
            "f1": f1_score(y_te, prev, pos_label="sim"), "auc": roc_auc_score(y_te == "sim", prob),
            "vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)}


buscas, resultados, tabela = {}, {}, []
for nome, (p, grade) in MODELOS.items():
    t0 = time.time()
    gs = GridSearchCV(p, grade, scoring=F1_SIM, cv=cv, n_jobs=-1, refit=True).fit(X_tr, y_tr)
    buscas[nome] = gs
    resultados[nome] = {"f1_cv": float(gs.best_score_), "melhores_parametros": {k.replace("clf__", ""): str(v) for k, v in gs.best_params_.items()},
                        "teste": metricas_teste(gs.best_estimator_)}
    for hp, valores in grade.items():
        tabela.append({"modelo": nome, "hiperparametro": hp.replace("clf__", ""),
                       "valores_testados": ", ".join(map(str, valores)), "melhor": str(gs.best_params_[hp])})
    print(f"{nome:20s} F1(cv)={gs.best_score_:.4f} | {time.time() - t0:.0f}s | {gs.best_params_}")

# ------------------------------------------------------------------ 4. escolha pela VALIDAÇÃO (não pelo teste)
melhor_nome = max(resultados, key=lambda n: resultados[n]["f1_cv"])
melhor = buscas[melhor_nome].best_estimator_
m = resultados[melhor_nome]["teste"]
print("Melhor pela validação cruzada:", melhor_nome, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})

# ------------------------------------------------------------------ 5. artefatos
os.makedirs("modelo", exist_ok=True)
joblib.dump(melhor, "modelo/modelo.joblib")                       # Pipeline completo (best_estimator_)
r3 = lambda v: round(float(v), 3)
vg = lambda v: f"{float(v):.3f}".replace(".", ",")          # número no formato brasileiro, para os textos do RAG
metricas = {
    "modelo": f"{melhor_nome} {resultados[melhor_nome]['melhores_parametros']}",
    "acuracia_teste": r3(m["acuracia"]), "precisao_teste": r3(m["precisao"]),
    "recall_teste": r3(m["recall"]), "especificidade_teste": r3(m["especificidade"]),
    "f1_teste": r3(m["f1"]), "auc_teste": r3(m["auc"]),
    "f1_validacao_cruzada": r3(resultados[melhor_nome]["f1_cv"]),
    "matriz_confusao_teste": {"verdadeiros_negativos": m["vn"], "falsos_positivos": m["fp"],
                              "falsos_negativos": m["fn"], "verdadeiros_positivos": m["vp"]},
    "n_treino": int(len(X_tr)), "n_teste": int(len(X_te)),
    "comparacao_modelos": {n: {"f1_validacao_cruzada": r3(r["f1_cv"]), "f1_teste": r3(r["teste"]["f1"]),
                               "recall_teste": r3(r["teste"]["recall"]), "auc_teste": r3(r["teste"]["auc"]),
                               "acuracia_teste": r3(r["teste"]["acuracia"])} for n, r in resultados.items()},
    "sklearn_version": sklearn.__version__,
}
json.dump(metricas, open("modelo/metricas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
pd.DataFrame(tabela).to_csv("modelo/tabela_hiperparametros.csv", index=False)

# amostra para conferir a PONTE legível -> modelo (ferramentas.conferir_transformacao)
amostra = [{"entrada": {k: (v.item() if hasattr(v, "item") else v) for k, v in linha.items()},
            "previsao_no_treino": str(p)}
           for (_, linha), p in zip(X_te.head(5).iterrows(), melhor.predict(X_te.head(5)))]
json.dump(amostra, open("modelo/amostra_conferencia.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# texto gerado para o RAG: nunca fica desatualizado em relação ao modelo salvo
linhas = ["# Modelo e métricas (texto gerado automaticamente por treinar.py)", "",
          "## Modelo escolhido",
          f"- O modelo do agente é {melhor_nome}, escolhido pelo maior F1 na validação cruzada de 5 dobras feita só no conjunto de treino. Parâmetros: {resultados[melhor_nome]['melhores_parametros']}.",
          "- O modelo é um Pipeline do scikit-learn: padronização das colunas numéricas (StandardScaler) e one-hot da escolaridade, tudo dentro do Pipeline, sem vazamento de dados do teste.",
          f"- O treino usou {len(X_tr)} clientes e o teste {len(X_te)} clientes (divisão 80/20 estratificada, random_state 42).",
          "- O modelo usa 8 atributos de entrada: limite de crédito, idade, escolaridade, situação do pagamento em setembro, agosto e julho, valor da fatura de setembro e valor pago em setembro.",
          "", "## Desempenho no conjunto de teste",
          f"- Acurácia {vg(metricas['acuracia_teste'])}, precisão {vg(metricas['precisao_teste'])}, recall (sensibilidade) {vg(metricas['recall_teste'])}, especificidade {vg(metricas['especificidade_teste'])}, F1 {vg(metricas['f1_teste'])} e AUC {vg(metricas['auc_teste'])}.",
          f"- Matriz de confusão no teste: {m['vn']} verdadeiros negativos, {m['fp']} falsos positivos, {m['fn']} falsos negativos e {m['vp']} verdadeiros positivos.",
          f"- Limitação principal: o recall é {vg(metricas['recall_teste'])}, ou seja, o modelo deixa de identificar parte dos clientes que realmente ficam inadimplentes. A acurácia alta engana porque a maioria dos clientes paga em dia.",
          "", "## Comparação dos três modelos"]
for n, r in metricas["comparacao_modelos"].items():
    linhas.append(f"- {n}: F1 na validação cruzada {vg(r['f1_validacao_cruzada'])}, F1 no teste {vg(r['f1_teste'])}, recall no teste {vg(r['recall_teste'])}, AUC no teste {vg(r['auc_teste'])}, acurácia no teste {vg(r['acuracia_teste'])}.")
Path("conhecimento/modelo_e_metricas.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
print(json.dumps(metricas, ensure_ascii=False, indent=1)[:1500])
