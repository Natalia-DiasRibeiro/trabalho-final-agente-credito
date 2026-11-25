"""
Dados e modelo do grupo: tudo o que é DETERMINÍSTICO (não usa o Gemini).

O agente (agente.py) decide QUANDO usar cada função; aqui fica O QUE cada uma faz:
  - preparar(entrada)            -> valida e converte o que o usuário informou (erros x alertas)
  - para_o_modelo(entrada)       -> a PONTE: formato legível -> formato do modelo
  - prever(entrada)              -> roda o modelo .joblib (classe + probabilidade)
  - casos_semelhantes(entrada)   -> busca clientes reais parecidos no dataset (validação)
  - estatisticas(coluna, grupo, filtro) -> números exatos do dataset (pandas)
  - fatos_do_dataset()           -> frases com números reais, geradas do CSV, para o RAG

Dataset: Default of Credit Card Clients (Yeh e Lien, 2009; clientes de cartão de Taiwan, abr-set/2005),
o mesmo do Trabalho Prático. Valores monetários em NT$ (novo dólar taiwanês).
Retornamos sempre tipos nativos (float, int, str, list, dict): numpy.int64 e NaN quebram o JSON.
"""
import json
import logging
import math
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
import sklearn

log = logging.getLogger(__name__)
BASE = Path(__file__).parent
MODELO = joblib.load(BASE / "modelo" / "modelo.joblib")
METRICAS = json.loads((BASE / "modelo" / "metricas.json").read_text(encoding="utf-8"))
DF = pd.read_csv(BASE / "dados" / "dataset_legivel.csv")
DF["faixa_etaria"] = pd.cut(DF["idade"], [0, 30, 40, 50, 60, 200],
                            labels=["até 30 anos", "31 a 40 anos", "41 a 50 anos", "51 a 60 anos", "mais de 60 anos"]).astype(str)

TAREFA = "classificacao"
ALVO = "inadimplente"                                   # classes: "sim" (inadimplente no mês seguinte) / "nao"
POSITIVA = "sim"
NUMERICAS = ["limite_credito", "idade", "situacao_pag_set", "situacao_pag_ago", "situacao_pag_jul",
             "fatura_set", "pagamento_set"]
CATEGORICAS = ["escolaridade"]
UNIDADE_ALVO = "se o cliente ficou inadimplente no mês seguinte: sim ou não"
OUTROS_GRUPOS = ["sexo", "estado_civil", "faixa_etaria"]   # só para estatísticas: NÃO são entradas do modelo

_PAG = ("situação do pagamento em {mes}: -2 = não usou o cartão, -1 = pagou a fatura integralmente, "
        "0 = pagou só o mínimo (crédito rotativo), 1 a 9 = meses de atraso")
# Descrição de cada atributo de entrada: o agente usa isto para extrair os valores da conversa
# e para perguntar o que falta. tipo: "numero", "inteiro" ou "texto".
ATRIBUTOS = {
    "limite_credito": {"tipo": "numero", "descricao": "limite de crédito do cartão, em NT$ (novo dólar taiwanês)"},
    "idade": {"tipo": "inteiro", "descricao": "idade do cliente em anos"},
    "escolaridade": {"tipo": "texto", "descricao": "escolaridade do cliente"},
    "situacao_pag_set": {"tipo": "inteiro", "descricao": _PAG.format(mes="setembro (o mês mais recente)")},
    "situacao_pag_ago": {"tipo": "inteiro", "descricao": _PAG.format(mes="agosto")},
    "situacao_pag_jul": {"tipo": "inteiro", "descricao": _PAG.format(mes="julho")},
    "fatura_set": {"tipo": "numero", "descricao": "valor da fatura de setembro, em NT$ (negativo se havia crédito a favor)"},
    "pagamento_set": {"tipo": "numero", "descricao": "valor que o cliente pagou em setembro, em NT$ (0 se não pagou nada)"},
}

if METRICAS.get("sklearn_version") != sklearn.__version__:
    log.error("ATENÇÃO: modelo salvo com scikit-learn %s, mas o servidor usa %s. Ajustem o requirements.txt.",
              METRICAS.get("sklearn_version"), sklearn.__version__)


def _norm(texto) -> str:
    """'  Pós-Graduação ' e 'pos graduacao' viram a mesma chave."""
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).casefold().replace("-", " ").strip()


def _nativo(v):
    """numpy.int64 -> int, numpy.float64 -> float (para virar JSON)."""
    return v.item() if hasattr(v, "item") else v


def _pct(x: float, casas: int = 1) -> str:
    """0.2213 -> '22,1%' (formato brasileiro, para os textos do RAG)."""
    return f"{x * 100:.{casas}f}".replace(".", ",") + "%"


def _dec(x: float, casas: int = 3) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def _num(x: float, casas: int = 2) -> str:
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


CATEGORIAS = {col: {_norm(v): _nativo(v) for v in DF[col].dropna().unique()}
              for col in CATEGORICAS + OUTROS_GRUPOS}
for col in CATEGORICAS:                                 # o agente mostra as opções válidas
    ATRIBUTOS[col]["valores"] = sorted(str(v) for v in CATEGORIAS[col].values())
FAIXAS = {col: (float(DF[col].min()), float(DF[col].max())) for col in NUMERICAS}
DESVIOS = {col: float(DF[col].std()) or 1.0 for col in NUMERICAS}
COLUNAS_CONSULTA = NUMERICAS + [ALVO]                   # ALVO -> contagem/proporção das classes
AGRUPAMENTOS = CATEGORICAS + OUTROS_GRUPOS + ["situacao_pag_set", ALVO]    # agrupar por ALVO -> médias por classe
FILTRAVEIS = NUMERICAS + CATEGORICAS + OUTROS_GRUPOS
OPERADORES = {">=": lambda s, v: s >= v, "<=": lambda s, v: s <= v, ">": lambda s, v: s > v,
              "<": lambda s, v: s < v, "==": lambda s, v: s == v}


# ---------------------------------------------------------------- PONTE: dataset legível -> modelo
# Existem DUAS versões dos dados:
#   - dados/dataset_legivel.csv: para PESSOAS e para o RAG (texto por extenso, números na escala original);
#   - o formato do MODELO: o que ele recebeu no treino (one-hot da escolaridade, números padronizados).
# Aqui usamos o "caso 1" do kit: o pré-processamento (StandardScaler + OneHotEncoder) está DENTRO do Pipeline
# salvo em modelo/modelo.joblib. Por isso a ponte só monta o DataFrame, com as colunas na ordem do treino,
# e o próprio Pipeline aplica as MESMAS transformações do treino (ajustadas só no conjunto de treino).
# conferir_transformacao() (abaixo) prova isso ao iniciar: refaz as previsões da amostra salva no treino.
def para_o_modelo(entrada: dict) -> pd.DataFrame:
    """Entrada legível (dict) -> DataFrame no formato que o modelo espera."""
    return pd.DataFrame([entrada])[NUMERICAS + CATEGORICAS]


def conferir_transformacao() -> dict:
    """Refaz as previsões da amostra salva no treino (modelo/amostra_conferencia.json) passando
    pela ponte para_o_modelo(). Se não baterem, a transformação está diferente da do treino."""
    arquivo = BASE / "modelo" / "amostra_conferencia.json"
    if not arquivo.exists():
        return {"ok": False, "detalhes": ["modelo/amostra_conferencia.json não existe: rodem treinar.py"]}
    problemas = []
    for item in json.loads(arquivo.read_text(encoding="utf-8")):
        try:
            obtido = _nativo(MODELO.predict(para_o_modelo(item["entrada"]))[0])
        except Exception as e:                     # ex.: colunas diferentes das do treino
            problemas.append(f"o modelo recusou a entrada transformada ({type(e).__name__}: {str(e)[:200]})")
            continue
        esperado = item["previsao_no_treino"]
        if str(obtido) != str(esperado):
            problemas.append(f"entrada {item['entrada']}: esperado {esperado}, obtido {obtido}")
    return {"ok": not problemas, "detalhes": problemas}


def faltando(entrada: dict) -> list[str]:
    """Atributos que o usuário ainda não informou."""
    return [a for a in ATRIBUTOS if entrada.get(a) in (None, "")]


def _regras_do_dominio(e: dict) -> list[str]:
    """Valores IMPOSSÍVEIS no mundo real -> erro (não prevê). O que só estiver fora da faixa do treino vira alerta."""
    # Só confere o que está em `e`: assim também serve para uma entrada PARCIAL (ver validar_parcial).
    erros = []
    if "limite_credito" in e and e["limite_credito"] <= 0:
        erros.append("o limite de crédito precisa ser maior que zero")
    if "idade" in e and not 18 <= e["idade"] <= 120:
        erros.append("a idade precisa estar entre 18 e 120 anos (titulares de cartão são adultos)")
    for col in ("situacao_pag_set", "situacao_pag_ago", "situacao_pag_jul"):
        if col in e and not -2 <= e[col] <= 9:
            erros.append(f"{col} precisa estar entre -2 e 9 (-2 não usou, -1 pagou tudo, 0 pagou o mínimo, 1 a 9 meses de atraso)")
    if "pagamento_set" in e and e["pagamento_set"] < 0:
        erros.append("o valor pago em setembro não pode ser negativo")
    return erros


def validar_parcial(entrada: dict) -> list[str]:
    """Valida só os atributos que o usuário JÁ informou (os que faltam são ignorados).
    Serve para avisar de um valor impossível (ex.: idade 250) mesmo quando ainda faltam outros dados."""
    erros, pronto = [], {}
    for nome, info in ATRIBUTOS.items():
        valor = entrada.get(nome)
        if valor in (None, ""):
            continue
        if info["tipo"] == "texto":
            if CATEGORIAS[nome].get(_norm(valor)) is None:
                erros.append(f"{nome} '{str(valor)[:40]}' não existe no dataset. Opções: {info['valores']}")
            continue
        try:
            x = _numero(valor)
        except (TypeError, ValueError):
            erros.append(f"{nome} precisa ser um número (recebido: {str(valor)[:40]!r})")
            continue
        if info["tipo"] == "inteiro" and not x.is_integer():
            erros.append(f"{nome} precisa ser um número inteiro (recebido: {str(valor)[:40]!r})")
            continue
        pronto[nome] = int(x) if info["tipo"] == "inteiro" else x
    return erros + _regras_do_dominio(pronto)


_MILHAR = r"-?[1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?"


def _numero(valor) -> float:
    """'80', 'NT$ 5.000', 'R$ 5.000,00', '2,5', '5 mil', '-1' -> float (formato brasileiro)."""
    s = str(valor)
    m = re.search(_MILHAR + r"(?![\d.])|-?\d+(?:[.,]\d+)?", s)
    if not m:
        raise ValueError(valor)
    t = m.group()
    t = t.replace(".", "").replace(",", ".") if re.fullmatch(_MILHAR, t) else t.replace(",", ".")
    x = float(t)
    if not math.isfinite(x):
        raise ValueError(valor)
    return x * 1000 if re.search(r"\bmil\b", s, re.I) else x


def preparar(entrada: dict) -> tuple[dict, list[str], list[str]]:
    """Converte tipos, corrige caixa/acento e separa ERROS (impossível) de ALERTAS (fora da faixa do treino)."""
    pronto, erros, alertas = {}, [], []
    for nome, info in ATRIBUTOS.items():
        valor = entrada.get(nome)
        if info["tipo"] == "texto":
            achado = CATEGORIAS[nome].get(_norm(valor))
            if achado is None:
                erros.append(f"{nome} '{str(valor)[:40]}' não existe no dataset. Opções: {info['valores']}")
            pronto[nome] = achado
            continue
        try:
            x = _numero(valor)
        except (TypeError, ValueError):
            erros.append(f"{nome} precisa ser um número (recebido: {str(valor)[:40]!r})")
            continue
        if info["tipo"] == "inteiro" and not x.is_integer():
            erros.append(f"{nome} precisa ser um número inteiro (recebido: {str(valor)[:40]!r})")
            continue
        pronto[nome] = int(x) if info["tipo"] == "inteiro" else x
    if erros:
        return pronto, erros, alertas
    erros += _regras_do_dominio(pronto)
    for col in NUMERICAS:
        mn, mx = FAIXAS[col]
        if not erros and not (mn <= pronto[col] <= mx):
            alertas.append(f"{col}={pronto[col]:.12g} está fora da faixa do dataset ({mn:.12g} a {mx:.12g}); a previsão é uma extrapolação")
    return pronto, erros, alertas


def prever(entrada: dict) -> dict:
    """Roda o modelo treinado com uma entrada já preparada (use preparar() antes)."""
    X = para_o_modelo(entrada)                          # ponte: legível -> formato do modelo
    classe = str(_nativo(MODELO.predict(X)[0]))
    classes = [str(_nativo(c)) for c in MODELO.classes_]
    probs = {c: round(float(p), 3) for c, p in zip(classes, MODELO.predict_proba(X)[0])}
    return {
        "previsao": classe,                              # "sim" = inadimplente no mês seguinte
        "probabilidades": probs,
        "probabilidade_inadimplencia_pct": round(probs[POSITIVA] * 100, 1),
        "regra_da_classe": "a classe prevista é a de maior probabilidade (limiar 0,5)",
        "acuracia_teste": float(METRICAS["acuracia_teste"]),
        "f1_teste": float(METRICAS["f1_teste"]),
        "recall_teste": float(METRICAS["recall_teste"]),
        "auc_teste": float(METRICAS["auc_teste"]),
    }


def casos_semelhantes(entrada: dict, previsao: str | None = None, prob_modelo_pct: float | None = None,
                      quantidade: int = 15) -> dict:
    """Clientes reais mais parecidos (mesma escolaridade, distância padronizada nas numéricas)."""
    base = DF[DF[CATEGORICAS[0]] == entrada[CATEGORICAS[0]]] if CATEGORICAS else DF
    if len(base) < quantidade:
        base = DF
    dist = sum(((base[c] - entrada[c]) / DESVIOS[c]) ** 2 for c in NUMERICAS) ** 0.5
    colunas = [c for c in DF.columns if c != "faixa_etaria"]
    proximos = base.assign(_d=dist).nsmallest(quantidade, "_d")[colunas]
    reais = proximos[ALVO].astype(str)
    contagem = {c: int(n) for c, n in reais.value_counts().items()}
    taxa = round(float((reais == POSITIVA).mean()) * 100, 1)
    taxa_geral = round(float((DF[ALVO] == POSITIVA).mean()) * 100, 1)
    resultado = {
        "quantidade_de_casos": int(len(proximos)),
        "casos": json.loads(proximos.to_json(orient="records", force_ascii=False)),
        "classes_dos_casos": contagem,
        "classe_mais_comum_nos_casos": max(contagem, key=contagem.get),
        "percentual_inadimplentes_nos_casos": taxa,
        "percentual_inadimplentes_no_dataset_todo": taxa_geral,
        "observacao": "os casos vêm do mesmo dataset usado no treino: a comparação é indicativa, não uma prova independente",
    }
    if previsao is not None:
        resultado["previsao_concorda_com_a_maioria"] = str(previsao) == resultado["classe_mais_comum_nos_casos"]
    if prob_modelo_pct is not None:
        dif = round(abs(prob_modelo_pct - taxa), 1)         # a conta é feita aqui, não pelo Gemini
        resultado["diferenca_em_pontos_percentuais"] = dif
        resultado["previsao_destoa_dos_casos_reais"] = bool(dif >= 20 or not resultado.get("previsao_concorda_com_a_maioria", True))
    return resultado


def _filtrar(filtro: dict | None) -> tuple[pd.DataFrame, str, str | None]:
    """Aplica um filtro simples {coluna, operador, valor}. Devolve (subconjunto, descrição, erro)."""
    if not filtro or not filtro.get("coluna"):
        return DF, "", None
    col, op, val = filtro.get("coluna"), str(filtro.get("operador", "")).strip(), filtro.get("valor", "")
    if col not in FILTRAVEIS:
        return DF, "", f"coluna de filtro inválida; opções: {FILTRAVEIS}"
    if op not in OPERADORES:
        return DF, "", f"operador inválido; opções: {list(OPERADORES)}"
    if col in CATEGORIAS:                                # categórica: só igualdade
        achado = CATEGORIAS[col].get(_norm(val))
        if achado is None:
            return DF, "", f"valor '{str(val)[:40]}' não existe em {col}; opções: {sorted(map(str, CATEGORIAS[col].values()))}"
        return DF[DF[col] == achado], f"{col} == {achado}", None
    try:
        x = _numero(val)
    except (TypeError, ValueError):
        return DF, "", f"o valor do filtro em {col} precisa ser um número"
    return DF[OPERADORES[op](DF[col], x)], f"{col} {op} {x:g}", None


def _estatisticas_em(df: pd.DataFrame, coluna: str, agrupar_por: str) -> dict:
    if coluna == ALVO:                                   # alvo é classe: conta em vez de tirar média
        classes = df[ALVO].astype(str)
        if agrupar_por and agrupar_por != ALVO:
            cont = pd.crosstab(df[agrupar_por], classes)
            prop = pd.crosstab(df[agrupar_por], classes, normalize="index")
            return {"coluna": ALVO, "por_grupo": [
                {agrupar_por: _nativo(g), "total": int(cont.loc[g].sum()),
                 "contagem": {str(c): int(cont.loc[g, c]) for c in cont.columns},
                 "proporcao": {str(c): round(float(prop.loc[g, c]), 4) for c in prop.columns}}
                for g in cont.index]}
        cont = classes.value_counts()
        return {"coluna": ALVO, "total": int(cont.sum()),
                "contagem_por_classe": {str(c): int(n) for c, n in cont.items()},
                "proporcao_por_classe": {str(c): round(float(n) / int(cont.sum()), 4) for c, n in cont.items()}}
    resumo = ["count", "mean", "median", "min", "max"]
    if agrupar_por:
        tabela = df.groupby(agrupar_por)[coluna].agg(resumo).round(2).reset_index()
        return {"coluna": coluna, "por_grupo": json.loads(tabela.to_json(orient="records", force_ascii=False))}
    serie = df[coluna].agg(resumo).round(2)
    return {"coluna": coluna, "resumo": {k: (None if pd.isna(v) else float(v)) for k, v in serie.items()}}


def estatisticas(coluna: str, agrupar_por: str = "", filtro: dict | None = None) -> dict:
    """Estatísticas REAIS (contagem, média, mediana, mín., máx. ou proporção das classes) de uma coluna,
    opcionalmente por grupo e/ou só para os clientes que atendem a um filtro (além do kit)."""
    if coluna not in COLUNAS_CONSULTA:
        return {"erro": f"coluna inválida; opções: {COLUNAS_CONSULTA}"}
    if agrupar_por and agrupar_por not in AGRUPAMENTOS:
        return {"erro": f"agrupamento inválido; opções: {AGRUPAMENTOS}"}
    df, descricao, erro = _filtrar(filtro)
    if erro:
        return {"erro": erro}
    if len(df) == 0:
        return {"filtro": descricao, "n_clientes_no_filtro": 0, "aviso": "nenhum cliente atende ao filtro"}
    resultado = _estatisticas_em(df, coluna, agrupar_por)
    if descricao:
        resultado = {"filtro": descricao, "n_clientes_no_filtro": int(len(df)), **resultado}
    return resultado


def fatos_do_dataset() -> list[str]:
    """Frases com números reais, geradas do CSV, para a base de conhecimento do RAG."""
    classes = DF[ALVO].astype(str)
    prop = classes.value_counts(normalize=True)
    pos = (classes == POSITIVA)
    fatos = [
        f"O dataset tem {_num(len(DF), 0)} clientes de cartão de crédito. A coluna-alvo é '{ALVO}' ({UNIDADE_ALVO}).",
        f"Proporção de clientes inadimplentes no dataset: {_pct(prop[POSITIVA], 2)}; adimplentes: {_pct(prop['nao'], 2)}. As classes são desbalanceadas.",
        f"O modelo do agente é o {METRICAS.get('modelo', '').split(' {')[0]}. No conjunto de teste: acurácia {_dec(METRICAS['acuracia_teste'])}, "
        f"F1 {_dec(METRICAS['f1_teste'])}, recall {_dec(METRICAS['recall_teste'])} e AUC {_dec(METRICAS['auc_teste'])}.",
    ]
    nomes = {"limite_credito": "limite de crédito (NT$)", "idade": "idade (anos)", "fatura_set": "fatura de setembro (NT$)",
             "pagamento_set": "valor pago em setembro (NT$)", "situacao_pag_set": "situação do pagamento em setembro",
             "situacao_pag_ago": "situação do pagamento em agosto", "situacao_pag_jul": "situação do pagamento em julho"}
    for col in NUMERICAS:
        s = DF[col]
        medias = DF.groupby(classes)[col].mean()
        fatos.append(f"A coluna '{col}' ({nomes[col]}) varia de {_num(s.min(), 0)} a {_num(s.max(), 0)}, com média {_num(s.mean())}; "
                     "média por classe: " + "; ".join(f"inadimplente = {c}: {_num(m)}" for c, m in medias.items()) + ".")
    for col in CATEGORICAS + OUTROS_GRUPOS:
        taxa = pos.groupby(DF[col]).mean().sort_values(ascending=False)
        qtd = DF[col].value_counts()
        fatos.append(f"Proporção de clientes inadimplentes por {col}, da maior para a menor: "
                     + "; ".join(f"{k}: {_pct(v)} ({_num(qtd[k], 0)} clientes)" for k, v in taxa.items()) + ".")
    grupos = DF.groupby("situacao_pag_set")[ALVO].agg(n="size", taxa=lambda s: (s == POSITIVA).mean())
    fatos.append("Proporção de clientes inadimplentes conforme a situação do pagamento em setembro (o mês mais recente; -2 não usou, "
                 "-1 pagou tudo, 0 pagou o mínimo, 1 a 9 meses de atraso): "
                 + "; ".join(f"situação {int(k)}: {_pct(r.taxa)} ({_num(r.n, 0)} clientes)" for k, r in grupos.iterrows() if r.n >= 30) + ".")
    corr = DF[NUMERICAS].corrwith(pos.astype(int)).sort_values(key=abs, ascending=False)
    fatos.append(f"Qual variável mais influencia a inadimplência? Correlação (r) de cada variável numérica com '{ALVO}' = '{POSITIVA}': "
                 + "; ".join(f"{k} r = {v:.2f}".replace(".", ",") for k, v in corr.items())
                 + f". A variável numérica mais associada à inadimplência é {corr.index[0]}.")
    return fatos


_conferencia = conferir_transformacao()
if not _conferencia["ok"]:
    log.error("ATENÇÃO: a ponte para_o_modelo() NÃO reproduz o treino: %s", _conferencia["detalhes"][:3])
