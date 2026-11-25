# Contexto do projeto (para o Claude que for trabalhar nesta pasta)

## O que é
Trabalho Final de Inteligência Artificial e Sistemas Inteligentes (UNOESC, 2026/2), grupo Natália Dias Ribeiro e Victor dos Santos Santin.
É um agente conversacional em Python (Flask) com RAG + um modelo de Machine Learning treinado no Trabalho Prático anterior
(dataset Default of Credit Card Clients, classificação de inadimplência). O motor de linguagem é a API do Gemini,
modelo `gemini-3.5-flash-lite`, plano GRATUITO (chave do Google AI Studio, conta Gmail pessoal autorizada pelo professor).
Leia o `README.md` desta pasta para a arquitetura (5 etapas: interpretar, recuperar, prever, gerar, verificar) e a seção
"O que mudamos em relação ao kit". O guia de tarefas e prazos está em `../GUIA_PASSO_A_PASSO.md`, se existir.

## Regras que valem sempre
- **A chave da API é secreta.** Nunca peça para o usuário colar a chave na conversa, nunca a escreva em código, em log ou no GitHub.
  Ela fica só no arquivo `.env` (já está no `.gitignore`) e, na hospedagem, em variável de ambiente `GEMINI_API_KEY`.
  Crie o `.env` a partir do `.env.example` e peça ao usuário para colar a chave NO ARQUIVO, por conta própria.
- **Não mudar o motor.** Só `gemini-3.5-flash-lite`, plano gratuito, chamadas diretas `generate_content`. Nada de function calling
  automático nem API de agentes do Gemini; o pipeline é do nosso código.
- **Prazos do professor:** checkpoint (link do app no ar) em 26/11; `links.txt` em 02/12 às 23h59. Depois dessa hora NÃO fazer commit
  na `main` nem deploy até a arguição (3/12). Se precisar mexer depois, só em outra branch.
- **Os dois integrantes fazem commits ao longo do trabalho** (o histórico vale nota). Não faça commits em nome de ninguém sem o usuário
  pedir; sugira commits pequenos e deixe o usuário confirmar. Nunca commitar `.env`.
- **Cada integrante precisa dominar o código** (arguição individual, com pequena alteração ao vivo). Explique o que fizer, em português,
  em linguagem simples.
- Número inventado zera teste da bateria: todo número da resposta do agente tem que vir do código (pandas, modelo ou base de conhecimento).

## Versões (o modelo .joblib foi treinado com elas; têm que bater)
Python 3.12 ou 3.13. `requirements.txt` fixa: scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5, google-genai 3.6.1, flask, gunicorn, python-dotenv, joblib.
Se trocar o Python ou o scikit-learn, rode `python treinar.py` de novo (~1,5 min) para regerar `modelo/`.

## Tarefa de configuração do ambiente (faça nesta ordem, explicando cada passo)
1. Verifique `python --version` (3.12 ou 3.13). Se for outra, oriente a instalar (python.org) antes de seguir.
2. Crie e ative o ambiente virtual (`python -m venv .venv`; Windows: `.venv\Scripts\activate`) e rode `pip install -r requirements.txt`.
3. Rode `python testes_automaticos.py`. Tem que terminar com "TUDO OK" (41 verificações, sem usar o Gemini).
4. Rode `python conferir.py`: a linha "Conferência da ponte legível -> modelo" tem que mostrar `ok: True`.
5. Copie `.env.example` para `.env` e peça ao usuário para abrir o arquivo e colar a chave em `GEMINI_API_KEY=` (sem aspas).
6. Teste a chave:
   `python -c "from dotenv import load_dotenv; load_dotenv(); from google import genai; print(genai.Client().models.generate_content(model='gemini-3.5-flash-lite', contents='Responda apenas: ok').text)"`
   Erros: 400/403 = chave ou região; 404 = modelo indisponível na conta; 429 = cota (esperar).
7. Rode `python app.py` e abra http://localhost:5000; teste 3 perguntas (uma de dados, uma previsão completa, uma fora do tema).
8. Rode `python registrar_testes.py` (gera `testes_registrados.md`) e `python comparar_sem_rag.py` (gera `comparacao_com_sem_rag.md`).
   Eles gastam cota do Gemini (~4 s entre chamadas). Mostre os resultados ao usuário e ajuste `prompts/*.md` ou o código se algo falhar.
9. Git/GitHub: `git init`, conferir com `git status` que `.env` NÃO aparece, e orientar o primeiro push (repositório público ou compartilhado com o professor).
10. Publicação (Render ou Streamlit Community Cloud): siga a seção "Publicar" do README. No Render, configure `PYTHON_VERSION` com a versão
    completa do Python usada no treino e a variável `GEMINI_API_KEY`. Abrir `/saude` para acordar o app.

## O que NÃO foi testado antes
Os prompts com o Gemini real (os testes automáticos usam um Gemini simulado), a versão Streamlit e o deploy. Tratar esses pontos como incertos.
