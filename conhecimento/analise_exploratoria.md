# Base de conhecimento: achados da análise exploratória

## Balanceamento das classes
Cerca de 22% dos clientes ficaram inadimplentes e 78% pagaram em dia. As classes são desbalanceadas, então a acurácia sozinha engana: um modelo que previsse que ninguém dá calote acertaria cerca de 78% dos clientes e não serviria para nada. Por isso o trabalho usa recall, F1 e AUC.

## Variável mais associada à inadimplência
A situação do pagamento é a variável mais associada ao alvo. A situação em setembro tem correlação de 0,32 com a inadimplência, a de agosto tem 0,26 e a de julho tem 0,24. Quanto mais antigo o mês, menor a correlação (0,22, 0,20 e 0,19 em junho, maio e abril).

## Atraso e risco
A inadimplência sobe muito com o atraso. Entre os clientes sem atraso, cerca de 13% a 17% ficaram inadimplentes; com um mês de atraso são 34%, e com dois meses sobe para 69%. Quem atrasou três meses ou mais tem risco ainda maior.

## Limite de crédito
O limite de crédito tem correlação negativa com a inadimplência (-0,15): clientes com limite maior tendem a pagar mais em dia. O limite médio de quem não ficou inadimplente é maior que o de quem ficou.

## Faturas, pagamentos e idade
As faturas dos seis meses são muito correlacionadas entre si (entre 0,80 e 0,95), e quase não se relacionam com a inadimplência. Os valores pagos têm correlação negativa fraca com o alvo. A idade quase não se relaciona com a inadimplência (0,01).

## Escolaridade, sexo e estado civil
A inadimplência varia pouco entre os grupos de escolaridade, sexo e estado civil, e essas variáveis têm correlação baixa com o alvo. Os grupos com poucos clientes, como Outros, têm taxas pouco confiáveis.

## Outliers
Os boxplots mostram muitos outliers em valores monetários, como faturas e pagamentos (cerca de 8% a 9% dos clientes pela regra do intervalo interquartil). Foram mantidos, porque representam clientes reais com faturas altas.

## Colunas de situação do pagamento são parecidas entre si
As colunas de situação de pagamento são correlacionadas entre si, porque um cliente que atrasou em um mês costuma continuar atrasado no seguinte. Isso explica por que os meses mais antigos acrescentam pouca informação ao modelo.
