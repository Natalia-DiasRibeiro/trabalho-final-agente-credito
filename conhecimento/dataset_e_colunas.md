# Base de conhecimento: o dataset e as colunas

## Sobre o dataset
O dataset é o Default of Credit Card Clients, de Yeh e Lien (2009), publicado pela UCI e no Kaggle. Ele reúne clientes de cartão de crédito de um banco de Taiwan, com dados de abril a setembro de 2005. É o mesmo dataset do Trabalho Prático de Aprendizado de Máquina deste grupo.

## O que cada linha representa
Cada linha é um cliente. As colunas descrevem o perfil do cliente, o histórico de pagamento de seis meses, o valor das faturas e o valor pago. O alvo indica se o cliente ficou inadimplente (não pagou) no mês seguinte, outubro de 2005.

## Tamanho e limpeza
O arquivo original tem 30.000 clientes. Foram removidas 35 linhas duplicadas exatas, restando 29.965 clientes. Valores de escolaridade fora do dicionário (0, 5 e 6) foram agrupados em "Outros", e o valor 0 de estado civil também virou "Outros". Os outliers foram mantidos, porque são clientes reais.

## Moeda
Os valores monetários estão em NT$, o novo dólar taiwanês. O dataset não usa reais.

## Colunas usadas como entrada do modelo
- limite_credito: limite de crédito do cartão em NT$, incluindo o crédito do titular e da família. Varia de 10.000 a 1.000.000.
- idade: idade do cliente em anos. Varia de 21 a 79.
- escolaridade: Pós-graduação, Universidade, Ensino médio ou Outros.
- situacao_pag_set: situação do pagamento da fatura em setembro de 2005, o mês mais recente. Na base original é a coluna PAY_0 (renomeada PAY_1).
- situacao_pag_ago: situação do pagamento em agosto de 2005 (PAY_2 na base original).
- situacao_pag_jul: situação do pagamento em julho de 2005 (PAY_3 na base original).
- fatura_set: valor da fatura de setembro de 2005 em NT$ (BILL_AMT1 na base original). Pode ser negativo quando há crédito a favor do cliente.
- pagamento_set: valor que o cliente pagou em setembro de 2005, em NT$ (PAY_AMT1 na base original).

## Como ler as colunas de situação do pagamento
- O valor -2 significa que o cliente não usou o cartão no mês.
- O valor -1 significa que o cliente pagou a fatura integralmente.
- O valor 0 significa que o cliente pagou apenas o mínimo e manteve o crédito rotativo.
- Os valores de 1 a 9 indicam quantos meses de atraso o cliente tem. Por exemplo, 2 significa atraso de dois meses.

## Colunas que existem no dataset original mas não entram no modelo
- sexo e estado_civil ficam só nas estatísticas. O sexo não é usado como entrada para evitar que o modelo discrimine por gênero.
- As situações de pagamento de abril, maio e junho (PAY_4, PAY_5 e PAY_6), as faturas de agosto a abril (BILL_AMT2 a BILL_AMT6) e os pagamentos de agosto a abril (PAY_AMT2 a PAY_AMT6) foram usados no Trabalho Prático, mas ficaram fora do agente.

## Por que 8 atributos
O enunciado do Trabalho Final permite no máximo 8 atributos de entrada. O subconjunto foi escolhido só com o conjunto de treino. A importância por permutação mostrou que a situação do pagamento em setembro, a fatura de setembro, o limite e os pagamentos recentes são os mais importantes. Idade e escolaridade foram mantidas para ter uma variável categórica e variáveis fáceis de informar. Na validação cruzada, esse subconjunto teve F1 de 0,476 e AUC de 0,765, contra F1 de 0,474 e AUC de 0,770 do modelo com as 23 colunas, ou seja, praticamente o mesmo desempenho.
