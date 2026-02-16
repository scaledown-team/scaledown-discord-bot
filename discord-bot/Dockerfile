# Build context must be the repo root (docs-main/), not discord-bot/
# Run: docker build -f discord-bot/Dockerfile -t scaledown-bot .

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY discord-bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source code
COPY discord-bot/healthcheck.py discord-bot/docs_loader.py \
     discord-bot/qa_engine.py discord-bot/bot.py ./

# Copy all documentation into /docs (App Runner has no volume mounts)
COPY README.md quickstart.mdx Documentation.mdx anatomy.mdx \
     development.mdx workflow_example.mdx rag_example.mdx \
     web-editor.mdx /docs/

COPY code_compression/ /docs/code_compression/
COPY generic-prompt-compression/ /docs/generic-prompt-compression/
COPY multimodal-audio-hallucination/ /docs/multimodal-audio-hallucination/

# Point the bot at the baked-in docs
ENV DOCS_PATH=/docs

# App Runner uses PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["python", "bot.py"]
