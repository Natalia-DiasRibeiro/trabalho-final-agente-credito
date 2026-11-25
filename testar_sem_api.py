"""Testa dados, modelo, RAG e verificação SEM gastar cota do Gemini:  python testar_sem_api.py
Genérico: monta os exemplos a partir da 1ª linha do CSV (funciona em regressão e classificação)."""
import os
os.environ.setdefault("GEMINI_API_KEY", "teste-offline")       # o Gemini não é chamado aqui

import ferramentas as F
from agente import verificar
from recuperacao import RECUPERADOR

print("Conferência da ponte legível -> modelo:", F.conferir_transformacao())
linha = F.DF.iloc[0]
exemplo = {a: str(F._nativo(linha[a])) for a in F.ATRIBUTOS}
print("0. Exemplo (1ª linha do CSV):", exemplo)
print("1. Faltando:", F.faltando({a: v for a, v in list(exemplo.items())[:-2]}))
entrada, erros, alertas = F.preparar(exemplo)
print("2. Preparar:", entrada, erros, alertas)
assert not erros, f"a 1ª linha do CSV deveria ser válida: {erros}"
p = F.prever(entrada)
print("3. Previsão:", p)
print("4. Casos semelhantes:", {k: v for k, v in F.casos_semelhantes(entrada, p["previsao"]).items() if k != "casos"})
num = F.NUMERICAS[0]
print("5. Fora da faixa (alerta):", F.preparar({**exemplo, num: str(F.FAIXAS[num][1] * 10 + 1)})[2])
if F.CATEGORICAS:
    print("6. Erro (categoria):", F.preparar({**exemplo, F.CATEGORICAS[0]: "Categoria-Que-Nao-Existe"})[1])
print("7. Estatísticas:", F.estatisticas(F.ALVO, F.AGRUPAMENTOS[0] if F.AGRUPAMENTOS else ""))
print("8. RAG:")
for pergunta in ["qual variável mais influencia?", "o modelo é confiável?", "receita de bolo"]:
    print("  ", pergunta, "->", [(t["similaridade"], t["trecho"][:60]) for t in RECUPERADOR.buscar(pergunta)])
print("9. Verificação OK  :", verificar(f"Métricas: {list(F.METRICAS.values())[1]}".replace(".", ","), {"m": F.METRICAS}, []))
print("10. Verificação FALHA:", verificar("O valor é 9.999,00", {"previsao_do_modelo": p}, []))
print("11. Teste do ERRO de domínio: escrevam aqui um caso IMPOSSÍVEL do seu domínio e confiram que vira erro.")
