# Liga Pause · Football Intelligence

App local/web para análise de Fantasy Liga Portugal: jogadores, liga privada,
dificuldade de calendário (FDR), previsão de pontos (xPts), otimizador de onze
e monitor de preços (Price Watch).

## Correr localmente

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Abre em `http://localhost:8501` (ou usa `--server.port 8502`).

## Deploy (Streamlit Community Cloud)

1. Este diretório é um repositório git próprio (isolado).
2. Publica-o no GitHub.
3. Em https://share.streamlit.io liga o repositório e aponta para `app.py`.
4. Não são necessários secrets — a app usa apenas a API pública da Liga Portugal.

## Notas

- Dados obtidos da API pública `fantasy.ligaportugal.pt`.
- A cache (`.cache/`) é regenerada da API e não é versionada.
- O FDR usa índices manuais em `calendar_ixd.json` / `ixd_data.json`.
- A primeira carga das previsões numa jornada nova é lenta (puxa históricos de
  todos os jogadores); depois fica em cache.
