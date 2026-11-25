"""
Recuperação (o "R" do RAG): encontra na base de conhecimento os trechos mais relacionados
com a pergunta, usando TF-IDF + similaridade de cosseno (scikit-learn, roda local e de graça).

Base de conhecimento = arquivos .md da pasta conhecimento/ (análise exploratória, descrição
das colunas, conclusões do trabalho anterior) + frases com números reais geradas do CSV
(ferramentas.fatos_do_dataset()).

Ideias para ir além do kit: dividir melhor os trechos, usar bigramas, ajustar o limiar,
combinar com busca por palavras-chave das colunas, medir a qualidade da busca com perguntas de teste.
"""
import re
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ferramentas import fatos_do_dataset

BASE = Path(__file__).parent
STOPWORDS_PT = """a o as os um uma uns umas de do da dos das no na nos nas em por para com sem sobre
e ou que se qual quais quanto quanta quantos quantas como onde quando é são foi ser ter tem há mais menos
muito pouco me te lhe isso isto esse essa este esta ao aos à às pelo pela pelos pelas meu minha seu sua""".split()
# sem acento, igual ao texto depois do strip_accents do TfidfVectorizer
STOPWORDS_PT = sorted({"".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c))
                       for w in STOPWORDS_PT})


# Além do kit: expansão da pergunta com palavras do domínio. O TF-IDF só casa palavras iguais; perguntas como
# "o modelo é confiável?" não contêm as palavras dos trechos certos ("limitações", "recall"). Cada regra liga
# um radical da pergunta (sem acento) às palavras-chave do trecho que a responde. Ajustada com avaliar_busca.py.
EXPANSOES = [
    (("confiavel", "confianca", "confiar", "funciona", "acerta", "erra", "bom ", "ruim", "boa "),
     "limitacoes recall deixa de identificar desempenho acuracia"),
    (("algoritmo", "comparad", "compara", "qual modelo", "modelos", "escolhid"), "comparacao dos tres modelos regressao logistica knn mlp"),
    (("melhoria", "melhorar", "melhorias", "aprimor", "futuro", "poderia ser feito"), "sugestoes de melhoria"),
    (("quantos", "quantas", "total de", "proporcao", "percentual", "porcentagem", "taxa de"), "proporcao de clientes inadimplentes dataset"),
    (("influencia", "importante", "mais pesa", "afeta", "relacao"), "correlacao variavel associada inadimplencia"),
]


def _sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto.casefold()) if not unicodedata.combining(c))


def expandir(pergunta: str) -> str:
    """Acrescenta à pergunta as palavras-chave dos trechos que costumam respondê-los."""
    p = _sem_acento(pergunta) + " "
    extras = [palavras for gatilhos, palavras in EXPANSOES if any(g in p for g in gatilhos)]
    return pergunta + " " + " ".join(extras)


def _trechos_dos_arquivos() -> list[str]:
    """Quebra cada .md em trechos: parágrafos e itens de lista viram trechos separados,
    e cada trecho leva o título da sua seção (ex.: "Limitações: o modelo não conhece...")."""
    trechos = []
    for arquivo in sorted((BASE / "conhecimento").glob("*.md")):
        secao = ""
        for linha_ou_bloco in re.split(r"\n\s*\n|\n(?=\s*[-*] )|\n(?=#)", arquivo.read_text(encoding="utf-8")):
            bloco = linha_ou_bloco.strip()
            if bloco.startswith("#"):                                     # título de seção
                titulo, _, resto = bloco.partition("\n")
                secao = titulo.strip("# ").strip()
                bloco = resto.strip()
            bloco = bloco.strip(" \n-*")
            if len(bloco) > 25:
                trechos.append(f"{secao}: {bloco}" if secao else bloco)
    return trechos


class Recuperador:
    def __init__(self):
        self.trechos = _trechos_dos_arquivos() + fatos_do_dataset()
        self.vetorizador = TfidfVectorizer(strip_accents="unicode", lowercase=True,
                                           stop_words=STOPWORDS_PT, ngram_range=(1, 2))
        self.matriz = self.vetorizador.fit_transform(self.trechos)

    def buscar(self, pergunta: str, k: int = 4, limiar: float = 0.05) -> list[dict]:
        """Devolve até k trechos com similaridade acima do limiar, do mais para o menos parecido."""
        if not pergunta.strip():
            return []
        notas = cosine_similarity(self.vetorizador.transform([expandir(pergunta)]), self.matriz)[0]
        ordem = notas.argsort()[::-1][:k]
        return [{"trecho": self.trechos[i], "similaridade": round(float(notas[i]), 3)}
                for i in ordem if notas[i] >= limiar]


RECUPERADOR = Recuperador()
