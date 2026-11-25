# Base de conhecimento: limitações e uso responsável

## O que a previsão significa
O modelo estima a probabilidade de um cliente ficar inadimplente no mês seguinte, com base em padrões do dataset. É uma estimativa estatística e não uma garantia, nem uma decisão de crédito.

## Recall baixo
O recall do modelo é baixo: ele deixa de identificar cerca de dois em cada três clientes que realmente ficam inadimplentes. Quando o modelo diz "não inadimplente", isso não garante que o cliente vai pagar.

## Probabilidade e classe
Para cada cliente o modelo calcula a probabilidade de inadimplência. A classe prevista é "sim" quando essa probabilidade passa de 50%. Uma probabilidade de 40% é uma classe "não", mas ainda indica risco acima da média do dataset, que é de cerca de 22%.

## Limiar de decisão
Baixar o limiar abaixo de 50% captura mais inadimplentes (aumenta o recall), em troca de mais alarmes falsos (reduz a precisão). Esse ajuste é uma sugestão de melhoria e não está ativo no agente. O ideal é escolher o limiar na validação, e não no conjunto de teste.

## Dados antigos e de um só país
Os dados são de clientes de Taiwan em 2005. O comportamento de clientes de outros países ou de épocas diferentes pode ser diferente, então o modelo não deve ser usado para decisões reais de crédito.

## Casos semelhantes
Os casos semelhantes que o agente mostra vêm do mesmo dataset usado no treino. A comparação é indicativa e não uma prova independente de que a previsão está certa.

## Atributos usados
O modelo usa só 8 atributos, enquanto o dataset original tem 23. O sexo não é usado como entrada. O desempenho com 8 atributos é praticamente igual ao do modelo com todas as colunas.

## Sugestões de melhoria
Tratar o desbalanceamento das classes (por exemplo com pesos de classe ou reamostragem), escolher o limiar na validação, testar outros algoritmos e calibrar as probabilidades.
