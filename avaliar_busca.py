"""
Mede a qualidade da busca do RAG (além do kit): para cada pergunta, o trecho certo aparece entre os k primeiros?

    python avaliar_busca.py            -> taxa de acerto e as perguntas que falharam
    python avaliar_busca.py varredura  -> testa combinações de k, limiar e n-gramas (como ajustamos a busca)

Cada item é (pergunta, texto que PRECISA aparecer em algum trecho recuperado).
"""
import os
import sys

os.environ.setdefault("GEMINI_API_KEY", "teste-offline")

from recuperacao import Recuperador  # noqa: E402

PERGUNTAS = [
    ("qual variável mais influencia a inadimplência?", "Correlação (r)"),
    ("esse modelo é confiável?", "deixa de identificar"),
    ("o que significa situação de pagamento 2?", "Os valores de 1 a 9"),
    ("quantos clientes são inadimplentes no total?", "Proporção de clientes inadimplentes no dataset"),
    ("por que usaram só 8 atributos?", "Por que 8 atributos"),
    ("em que moeda estão os valores?", "NT$"),
    ("de onde e de que ano são os dados?", "Taiwan"),
    ("qual o desempenho do modelo no teste?", "Desempenho no conjunto de teste"),
    ("as classes são desbalanceadas?", "desbalanceadas"),
    ("o que é o limiar de decisão?", "Limiar de decisão"),
    ("o sexo é usado pelo modelo?", "O sexo não é usado"),
    ("quais algoritmos foram comparados?", "Comparação dos três modelos"),
    ("como o atraso afeta o risco de calote?", "Atraso e risco"),
    ("que melhorias poderiam ser feitas?", "Sugestões de melhoria"),
    ("o que são os casos semelhantes?", "Casos semelhantes"),
    ("o limite de crédito influencia a inadimplência?", "Limite de crédito"),
]

# Perguntas NOVAS, escritas depois de ajustar a busca (conjunto de validação: não foram usadas para ajustar):
PERGUNTAS_NOVAS = [
    ("o modelo acerta sempre?", "deixa de identificar"),
    ("o que fazer para melhorar o modelo?", "Sugestões de melhoria"),
    ("qual a escala dos valores da situação de pagamento?", "Os valores de 1 a 9"),
    ("por que as faturas não entram no modelo?", "Colunas que existem no dataset original"),
    ("qual a taxa de calote por escolaridade?", "por escolaridade"),
    ("como foi feita a limpeza dos dados?", "Tamanho e limpeza"),
]


def taxa(rec: Recuperador, k: int, limiar: float, detalhar=False, perguntas=None) -> float:
    perguntas = perguntas or PERGUNTAS
    acertos, falhas = 0, []
    for pergunta, esperado in perguntas:
        trechos = [t["trecho"] for t in rec.buscar(pergunta, k=k, limiar=limiar)]
        if any(esperado.casefold() in t.casefold() for t in trechos):
            acertos += 1
        else:
            falhas.append(pergunta)
    if detalhar:
        for f in falhas:
            print("  FALHOU:", f)
    return acertos / len(perguntas)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "varredura":
        for ngram in [(1, 1), (1, 2)]:
            for k in [3, 4, 5, 6]:
                for limiar in [0.03, 0.05, 0.10]:
                    r = Recuperador(); r.vetorizador.set_params(ngram_range=ngram)
                    r.matriz = r.vetorizador.fit_transform(r.trechos)
                    print(f"ngram={ngram} k={k} limiar={limiar}: acerto={taxa(r, k, limiar):.2f}")
    else:
        from recuperacao import RECUPERADOR
        print(f"Acerto da busca (ajuste): {taxa(RECUPERADOR, 4, 0.05, detalhar=True):.2f} ({len(PERGUNTAS)} perguntas, k=4)")
        print(f"Acerto da busca (perguntas novas): {taxa(RECUPERADOR, 4, 0.05, detalhar=True, perguntas=PERGUNTAS_NOVAS):.2f} ({len(PERGUNTAS_NOVAS)} perguntas)")
