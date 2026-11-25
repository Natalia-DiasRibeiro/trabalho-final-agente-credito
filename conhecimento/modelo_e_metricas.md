# Modelo e métricas (texto gerado automaticamente por treinar.py)

## Modelo escolhido
- O modelo do agente é MLP, escolhido pelo maior F1 na validação cruzada de 5 dobras feita só no conjunto de treino. Parâmetros: {'activation': 'relu', 'alpha': '0.01', 'hidden_layer_sizes': '(64,)'}.
- O modelo é um Pipeline do scikit-learn: padronização das colunas numéricas (StandardScaler) e one-hot da escolaridade, tudo dentro do Pipeline, sem vazamento de dados do teste.
- O treino usou 23972 clientes e o teste 5993 clientes (divisão 80/20 estratificada, random_state 42).
- O modelo usa 8 atributos de entrada: limite de crédito, idade, escolaridade, situação do pagamento em setembro, agosto e julho, valor da fatura de setembro e valor pago em setembro.

## Desempenho no conjunto de teste
- Acurácia 0,819, precisão 0,688, recall (sensibilidade) 0,336, especificidade 0,957, F1 0,452 e AUC 0,751.
- Matriz de confusão no teste: 4465 verdadeiros negativos, 202 falsos positivos, 880 falsos negativos e 446 verdadeiros positivos.
- Limitação principal: o recall é 0,336, ou seja, o modelo deixa de identificar parte dos clientes que realmente ficam inadimplentes. A acurácia alta engana porque a maioria dos clientes paga em dia.

## Comparação dos três modelos
- Regressão Logística: F1 na validação cruzada 0,352, F1 no teste 0,364, recall no teste 0,244, AUC no teste 0,713, acurácia no teste 0,812.
- KNN: F1 na validação cruzada 0,461, F1 no teste 0,454, recall no teste 0,354, AUC no teste 0,733, acurácia no teste 0,811.
- MLP: F1 na validação cruzada 0,483, F1 no teste 0,452, recall no teste 0,336, AUC no teste 0,751, acurácia no teste 0,819.
