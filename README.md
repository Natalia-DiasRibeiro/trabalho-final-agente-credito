# Assistente de Risco de Crédito: agente com RAG + modelo de ML

Trabalho Final de Inteligência Artificial e Sistemas Inteligentes (UNOESC, 2026/2).
Grupo: Natália Dias Ribeiro e Victor dos Santos Santin.

Agente conversacional, escrito em Python, que conversa sobre o dataset do Trabalho Prático (**Default of Credit Card Clients**, Yeh e Lien, 2009: 29.965 clientes de cartão de Taiwan, após limpeza) e estima a probabilidade de um cliente ficar inadimplente no mês seguinte. O Gemini (`gemini-3.5-flash-lite`, plano gratuito) só **interpreta** a pergunta e **redige** a resposta. Buscar nos dados, rodar o modelo e conferir os números é código nosso.

> ⚠️ É um trabalho acadêmico. A previsão é uma estimativa estatística sobre dados de Taiwan de 2005, **não** uma decisão de crédito.

## Como o agente funciona

```
mensagem → página de chat → servidor Flask (app.py) → agente.responder()
   1 INTERPRETAR (Gemini → JSON)   intenção + atributos citados + consulta (coluna, grupo, filtro) + busca
   2 RECUPERAR   (nosso código)    RAG: TF-IDF na base de conhecimento + estatísticas exatas com pandas
   3 PREVER      (nosso código)    o que falta? → pergunta | valida (erro x alerta) → para_o_modelo() → modelo.joblib
                                   → casos reais semelhantes (comparação com a previsão)
   4 GERAR       (Gemini)          resposta usando SÓ o CONTEXTO das etapas 2 e 3
   5 VERIFICAR   (nosso código)    todo número da resposta está no CONTEXTO? senão: refaz 1 vez ou avisa
```

O Gemini nunca vê o CSV nem o modelo. Usamos `generate_content` direto (sem function calling automático e sem a API de agentes). A página mostra as etapas de cada resposta (🔧).

## Estrutura

```
app.py                  servidor Flask: página + /api/chat + /saude (limites de mensagens, chave só no servidor)
versao_streamlit/       a mesma interface em Streamlit (alternativa de hospedagem)
agente.py               o pipeline de 5 etapas
ferramentas.py          dados e modelo: preparar, para_o_modelo (ponte), prever, casos_semelhantes, estatisticas, fatos
recuperacao.py          RAG: TF-IDF + expansão da pergunta com palavras do domínio
conhecimento/           base de conhecimento (4 arquivos .md; cada parágrafo ou item vira um trecho)
prompts/                interpretar.md (etapa 1) e responder.md (etapa 4)
modelo/                 modelo.joblib (Pipeline completo), metricas.json, amostra_conferencia.json, tabela_hiperparametros.csv
dados/                  dataset_legivel.csv (para o agente) e o CSV original do dataset (para treinar.py)
treinar.py              limpeza + treino + escolha do modelo + geração dos artefatos
conferir.py             gabarito (estatísticas e previsões) a partir do CSV e do modelo do repositório
testes_automaticos.py   41 verificações sem Gemini (ferramentas + pipeline com Gemini simulado)
avaliar_busca.py        mede a qualidade da busca do RAG
comparar_sem_rag.py     mesmas perguntas com e sem RAG (para o relatório)
registrar_testes.py     roda 12 testes no agente real e salva o registro (para o relatório)
```

## Como rodar

Python **3.12 ou 3.13** (3.11 não instala o numpy fixado).

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python treinar.py                   # opcional: refaz o modelo (já vem pronto em modelo/); ~1,5 min
python testes_automaticos.py        # testes sem Gemini
python conferir.py                  # gabarito do professor
cp .env.example .env                # Windows: copy .env.example .env  → coloque a GEMINI_API_KEY
python app.py                       # http://localhost:5000
```

A chave vem do Google AI Studio (conta UNOESC, plano gratuito, sem faturamento). **Nunca** vai para o front-end nem para o GitHub (o `.env` está no `.gitignore`). Na hospedagem, ela é uma variável de ambiente (`GEMINI_API_KEY`).

## O modelo

Pipeline do scikit-learn (`StandardScaler` nas numéricas + `OneHotEncoder` na escolaridade, **dentro** do Pipeline) salvo com joblib. É o mesmo problema do Trabalho Prático, com as mesmas etapas: duplicatas removidas, split 80/20 estratificado com `random_state=42`, GridSearchCV de 5 dobras com F1, e os três modelos (Regressão Logística, KNN e MLP) comparados.

- **8 atributos de entrada** (o limite do enunciado): `limite_credito`, `idade`, `escolaridade`, `situacao_pag_set`, `situacao_pag_ago`, `situacao_pag_jul`, `fatura_set` e `pagamento_set`. O dataset original tem 23. O subconjunto foi escolhido só com o conjunto de treino (importância por permutação). Na validação cruzada, ele teve F1 de 0,476 e AUC de 0,765, contra 0,474 e 0,770 com as 23 colunas. O sexo não é entrada do modelo (evita discriminação por gênero).
- **Escolha pelo F1 da validação cruzada** (e não pelo teste): MLP `(64,)`, relu, alpha 0,01.
- **Desempenho no teste** (5.993 clientes): acurácia 0,819, precisão 0,688, recall 0,336, especificidade 0,957, F1 0,452 e AUC 0,751. O recall é baixo: o modelo deixa passar a maior parte dos inadimplentes. O agente diz isso ao usuário.
- **Ponte `para_o_modelo()`:** como o pré-processamento está dentro do Pipeline, a ponte monta o DataFrame na ordem do treino e o Pipeline aplica as transformações ajustadas no treino. Ao iniciar, o app refaz as previsões de `modelo/amostra_conferencia.json` e avisa no log se não baterem (`conferir_transformacao()`).

## O que mudamos em relação ao kit

**Troca do exemplo pelo nosso trabalho (o que o enunciado pede)**
- Dataset, modelo, `ATRIBUTOS`, regras de validação do domínio (idade entre 18 e 120, limite maior que zero, situação de pagamento entre -2 e 9, pagamento não negativo), prompts, base de conhecimento e textos da página.
- `treinar.py` novo: limpeza igual à do Trabalho Prático, treino dos 3 modelos e geração de `modelo.joblib`, `metricas.json`, `amostra_conferencia.json` e do texto de métricas para o RAG (`conhecimento/modelo_e_metricas.md`), que nunca fica desatualizado em relação ao modelo.

**Além do kit**
1. **Escolha de atributos com método:** importância por permutação e validação cruzada só no treino, comparando subconjuntos de 8 atributos com o modelo de 23 (justificativa no relatório). A escolha do melhor modelo passou a ser pela validação cruzada, não pelo teste.
2. **Consultas novas aos dados:** filtro (`coluna`, `operador`, `valor`) no esquema da etapa 1 e em `estatisticas()` (ex.: proporção de inadimplentes entre quem atrasou 2 ou mais meses), mais agrupamentos por faixa etária, sexo, estado civil e situação do pagamento.
3. **Casos semelhantes melhores:** 15 vizinhos reais, com o percentual de inadimplentes entre eles, a diferença em pontos percentuais para a probabilidade do modelo e um sinal `previsao_destoa_dos_casos_reais`, tudo calculado no código (o Gemini não faz contas).
4. **RAG ajustado e medido:** base de conhecimento dividida por tema, fatos gerados do CSV (proporções por grupo, por situação de pagamento, correlações) e **expansão da pergunta com palavras do domínio** (`recuperacao.EXPANSOES`). Com `avaliar_busca.py`, a busca acerta o trecho certo em 81% de 16 perguntas sem a expansão e em 100% com ela. Em 6 perguntas escritas depois do ajuste, vai de 50% para 100%.
5. **Prompts reescritos para classificação:** probabilidade, métricas reais (recall baixo), comparação com casos reais, honestidade sobre os limites e regras de conversão da entrada em linguagem natural (ex.: "pagou só o mínimo" = 0, "atrasou 2 meses" = 2, "uns 50 mil" = 50000).
6. **Testes automáticos próprios** (`testes_automaticos.py`, 41 verificações): validação, ponte, estatísticas, verificação de números, busca do RAG e o pipeline completo com um Gemini simulado, cobrindo as 12 categorias da bateria. Eles testam a lógica do código, não a qualidade dos prompts.
7. **Registro de testes e comparação com × sem RAG** automatizados contra o Gemini real (`registrar_testes.py`, `comparar_sem_rag.py`).
8. **Página:** exemplos de perguntas clicáveis e aviso de "acordando o servidor" quando a resposta demora (plano gratuito do Render).

## Segurança

- Chave em variável de ambiente; nunca no front-end nem no GitHub.
- Limite de mensagens por IP e por dia (`app.py`), tamanho máximo de mensagem e de requisição.
- Etapa 1 classifica assuntos fora do tema e tentativas de "quebrar as regras" como `fora_escopo`, e a resposta é fixa, sem chamar o Gemini de novo. O prompt de resposta também ignora instruções que venham dentro do contexto ou das mensagens.
- A página usa `textContent` (sem injeção de HTML).

## Publicar

Render (recomendado) ou Streamlit Community Cloud, como no kit:
- **Render:** New → Web Service → repositório; build `pip install -r requirements.txt`; start `gunicorn app:app --workers 1 --threads 4 --timeout 120`; variáveis `GEMINI_API_KEY` e `PYTHON_VERSION` (a versão completa do Python do treino, ex.: 3.13.x). Abra `/saude` antes da apresentação para acordar o app.
- **Streamlit:** arquivo principal `versao_streamlit/streamlit_app.py`; chave em Secrets; Python 3.12 ou 3.13.

De 02/12 às 23h59 até a arguição: nenhum commit na `main` e nenhum deploy.

## Limitações

- Os dados são de Taiwan em 2005: o modelo não serve para decisões reais.
- O recall do modelo é baixo (0,336): ele deixa de identificar cerca de dois em cada três inadimplentes.
- Os "casos semelhantes" vêm do mesmo dataset do treino: a validação é indicativa, não independente.
- A verificação confere os **números** da resposta, não se o número está atribuído ao grupo certo nem as categorias citadas.
- O limiar de decisão é 0,5. Ajustar o limiar na validação é uma melhoria futura.
