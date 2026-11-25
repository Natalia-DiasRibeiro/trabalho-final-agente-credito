"""
Servidor: página de chat + rota /api/chat, que chama o nosso agente (agente.py).

A chave do Gemini fica SÓ aqui no servidor (variável de ambiente GEMINI_API_KEY).

Rodar localmente:  crie o .env com GEMINI_API_KEY=... e rode  python app.py  -> http://localhost:5000
Em produção (Render):  gunicorn app:app --workers 1 --threads 4 --timeout 120
"""
import logging
import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()                                    # lê o .env, se existir (uso local)
_chave = os.environ.get("GEMINI_API_KEY", "")
if not _chave or _chave.startswith("cole-aqui"):
    raise RuntimeError("Defina a variável de ambiente GEMINI_API_KEY com a sua chave (arquivo .env ou painel da hospedagem).")

from google.genai import errors                  # noqa: E402
import agente                                    # noqa: E402

logging.basicConfig(level=logging.INFO)
MAX_CARACTERES = 1000                            # tamanho máximo de uma mensagem
LIMITE_POR_IP = (40, 600)                        # no máx. 40 mensagens a cada 10 min por IP
LIMITE_GLOBAL = (150, 86400)                     # no máx. 150 mensagens/dia no app todo (~300-450 chamadas: protege a cota)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024     # rejeita requisições maiores que 64 KB
_acessos: dict[str, deque] = defaultdict(deque)


def _excedeu(chave: str, limite: tuple[int, int]) -> bool:
    """Limite de melhor esforço, em memória (zera se o servidor reiniciar)."""
    maximo, janela = limite
    agora, fila = time.time(), _acessos[chave]
    while fila and agora - fila[0] > janela:
        fila.popleft()
    if len(fila) >= maximo:
        return True
    fila.append(agora)
    return False


@app.get("/")
def pagina():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/saude")
def saude():
    """Rota leve para "acordar" o app antes da apresentação."""
    return jsonify(ok=True, modelo=agente.MODELO_GEMINI)


@app.post("/api/chat")
def chat():
    xff = request.headers.get("X-Forwarded-For", "")
    ip = xff.split(",")[-1].strip() if xff else (request.remote_addr or "?")
    if _excedeu("__global__", LIMITE_GLOBAL):
        return jsonify(erro="O app atingiu o limite diário de mensagens. Tente amanhã."), 429
    if _excedeu(ip, LIMITE_POR_IP):
        return jsonify(erro="Muitas mensagens em pouco tempo. Aguarde alguns minutos."), 429

    dados = request.get_json(silent=True)
    if not isinstance(dados, dict):
        return jsonify(erro="Requisição inválida."), 400
    mensagem = str(dados.get("mensagem", "")).strip()[:MAX_CARACTERES]
    if not mensagem:
        return jsonify(erro="Mensagem vazia."), 400
    historico = dados.get("historico")
    historico = [h for h in historico if isinstance(h, dict)] if isinstance(historico, list) else []

    try:
        resultado = agente.responder(mensagem, historico)
    except errors.APIError as e:
        app.logger.warning("Erro do Gemini: %s", e)
        if getattr(e, "code", None) == 429:
            return jsonify(erro="Limite gratuito do Gemini atingido. Tente de novo em instantes."), 503
        return jsonify(erro=f"O agente está indisponível no momento (Gemini {getattr(e, 'code', '?')})."), 503
    except Exception as e:
        app.logger.exception("Falha inesperada no agente")
        if any(k in type(e).__name__ for k in ("Timeout", "Connect", "Proxy", "Network")):
            return jsonify(erro="Falha de rede ao falar com o Gemini. Tente de novo em instantes."), 503
        return jsonify(erro="Erro interno no agente. Veja o log do servidor."), 500

    app.logger.info("pergunta=%r etapas=%s", mensagem[:80], resultado["etapas"])
    return jsonify(resposta=resultado["resposta"], etapas=resultado["etapas"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
