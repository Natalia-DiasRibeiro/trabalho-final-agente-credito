"""
Comparação para o relatório: as MESMAS perguntas respondidas
  (a) pelo Gemini "puro", só com a persona (sem RAG, sem modelo, sem verificação), e
  (b) pelo agente completo do grupo.

    python comparar_sem_rag.py            (usa a chave do .env; gasta ~3 chamadas por pergunta)

Saída: no terminal e em comparacao_com_sem_rag.md (cole os trechos no relatório e comente as diferenças).
"""
import time

from dotenv import load_dotenv

load_dotenv()
from google.genai import types  # noqa: E402

import agente  # noqa: E402

PERGUNTAS = [
    "Qual a proporção de clientes inadimplentes por escolaridade?",
    "Qual a taxa de inadimplência de quem atrasou 2 ou mais meses em setembro?",
    "Qual a probabilidade de inadimplência de um cliente de 35 anos, pós-graduação, limite de 50000, atraso de 2 meses em "
    "setembro, 1 mês em agosto, pagou tudo em julho, fatura de setembro de 30000 e pagamento de setembro de 0?",
    "Qual variável mais influencia a inadimplência?",
    "Esse modelo é confiável?",
]

saida = ["# Comparação com × sem RAG (gerado por comparar_sem_rag.py)", ""]
for p in PERGUNTAS:
    puro = agente._gemini(agente.INSTRUCOES, [types.Content(role="user", parts=[types.Part(text=p)])])
    completo = agente.responder(p, [])
    bloco = ["=" * 80, f"PERGUNTA: {p}", "", "[SEM RAG]", puro, "",
             "[AGENTE COMPLETO]  etapas: " + " · ".join(completo["etapas"]), completo["resposta"], ""]
    print("\n".join(bloco))
    saida += [f"## {p}", "", "**Sem RAG (Gemini puro):**", "", puro, "",
              "**Agente completo** (etapas: " + " · ".join(completo["etapas"]) + "):", "", completo["resposta"], ""]
    time.sleep(3)                                   # respeita o limite por minuto do plano gratuito
open("comparacao_com_sem_rag.md", "w", encoding="utf-8").write("\n".join(saida))
print("Salvo em comparacao_com_sem_rag.md")
