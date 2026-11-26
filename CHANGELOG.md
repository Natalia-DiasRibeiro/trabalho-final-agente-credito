# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e [SemVer](https://semver.org/lang/pt-BR/).

## [1.1.0] - 2026-11-25

### Adicionado
- Interface do chat redesenhada em tema claro: fundo de pontos que reage suavemente ao cursor, faixa do pipeline que acompanha a resposta, efeito de digitação, destaque de números, rastro das etapas por resposta e indicador de status do servidor.
- Sugestões de perguntas na tela inicial, separadas em Dados, Prever um cliente e Sobre o modelo.
- Layout responsivo para celular e computador (altura dinâmica, áreas seguras do iPhone, campo que cresce com o texto, Enter envia e Shift+Enter quebra linha) e respeito a `prefers-reduced-motion`.
- Template de Pull Request do GitHub.

## [1.0.0] - 2026-11-25

### Adicionado
- Agente de risco de crédito publicado no Render, com pipeline próprio de 5 etapas (interpretar, recuperar, prever, gerar, verificar), RAG com TF-IDF e expansão de consulta, modelo MLP do Trabalho Prático, validação parcial da entrada e verificação de números.
