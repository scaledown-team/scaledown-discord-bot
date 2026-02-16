# ScaleDown Discord Documentation Bot — Setup Guide

A Discord bot that answers questions about the entire ScaleDown platform — quickstart, REST API, Python SDK, RAG pipelines, code compression, hallucination reduction, and more — using ScaleDown's own query-aware compression API to retrieve and compress relevant documentation before generating answers.

## Architecture

```
User asks question in Discord (/ask, !ask, or @mention)
        ↓
  docs_loader.py — loads all .mdx/.md docs, preserves structure,
                   strips MDX component tags
        ↓
  qa_engine.py  — chunks the knowledge base by section
                → compresses each chunk via ScaleDown REST API
                  (query-aware: keeps what's relevant, removes what isn't)
                → combines compressed chunks
                → sends to LLM with a comprehensive ScaleDown system prompt
        ↓
  bot.py — sends the answer back to Discord
```

The bot has deep, holistic knowledge of ScaleDown baked into its system prompt covering:
- Getting started & first API call
- REST API (Python, TypeScript, JavaScript)
- Python SDK (`ScaleDownCompressor`, `HasteOptimizer`, `SemanticOptimizer`, `Pipeline`)
- All data structures and configuration
- RAG without a vector database
- Code compression (AST-based HASTE)
- Pareto Merging (reasoning optimization)
- Hallucination reduction pipeline (Baseline → APO → CoVe → Self-Correction)
- Modular prompt optimization framework
- Prompt anatomy & best practices
- Supported models & error handling

## Prerequisites

- Python 3.10+
- A Discord account with a server you manage
- API keys for: Discord Bot, ScaleDown, OpenAI

## Step 1: Create a Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** → name it `ScaleDown Docs Bot` → **Create**
3. Go to the **Bot** tab:
   - Click **Reset Token** → copy the token (you'll need it for `.env`)
   - Under **Privileged Gateway Intents**, enable **Message Content Intent**
4. Go to the **OAuth2** tab:
   - Under **Scopes**, select `bot` and `applications.commands`
   - Under **Bot Permissions**, select:
     - Send Messages
     - Send Messages in Threads
     - Read Message History
     - Use Slash Commands
     - Add Reactions
   - Copy the generated URL and open it in your browser to invite the bot to your server

## Step 2: Create a Channel (Optional)

1. In your Discord server, create a text channel called `#scaledown-support`
2. Set `BOT_CHANNEL_NAME=scaledown-support` in your `.env` to restrict the bot to that channel
3. If you leave `BOT_CHANNEL_NAME` empty, the bot responds in any channel

## Step 3: Install Dependencies

```bash
cd discord-bot
pip install -r requirements.txt
```

Dependencies are lightweight — no `scaledown` Python package needed since the bot uses the REST API directly:
- `discord.py` — Discord bot framework
- `openai` — LLM answer generation
- `requests` — ScaleDown REST API calls
- `python-dotenv` — Environment variable loading

## Step 4: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `DISCORD_BOT_TOKEN` | Bot token from Step 1 |
| `SCALEDOWN_API_KEY` | Your ScaleDown API key ([get one here](https://blog.scaledown.ai/blog/getting-started)) |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `DOCS_PATH` | Path to the docs folder (default: `../`) |
| `LLM_MODEL` | LLM model to use (default: `gpt-4o`) |
| `BOT_CHANNEL_NAME` | Restrict to this channel (optional) |

## Step 5: Run the Bot

```bash
python bot.py
```

You should see:

```
[docs_loader] Loaded 11 documents (42,500 chars, ~10,625 tokens est.)
[qa_engine] Ready — 15 doc chunks, LLM: gpt-4o
[bot] Logged in as ScaleDown Docs Bot#1234 (ID: ...)
[bot] Synced 3 slash command(s)
[bot] Ready!
```

## Usage

### Commands

| Command | Description |
|---|---|
| `/ask <question>` | Ask anything about ScaleDown (recommended) |
| `/quickstart` | Get the full quickstart guide with first API call |
| `/examples` | See common usage examples (REST API, SDK, RAG, HASTE) |
| `!ask <question>` | Prefix alternative for /ask |
| `!quickstart` | Prefix alternative for /quickstart |
| `!examples` | Prefix alternative for /examples |
| `@ScaleDown Docs Bot <question>` | Mention the bot with a question |
| `!ping` | Check if the bot is online |
| `!help_sd` | Show all commands and topics |

### Example Questions

**Getting started:**
- "How do I get started with ScaleDown?"
- "What do I need to make my first API call?"

**REST API:**
- "Show me how to compress a prompt with the REST API in Python"
- "What does the API response look like?"

**Python SDK:**
- "What parameters does ScaleDownCompressor accept?"
- "How do I use preserve_keywords?"
- "What's the difference between rate=auto and a fixed rate?"

**RAG:**
- "How do I build RAG without a vector database?"
- "How does ScaleDown's query-aware compression work?"

**Code compression:**
- "What is the HasteOptimizer and when should I use it?"
- "How does AST-based code compression work?"

**Advanced features:**
- "How does the hallucination reduction pipeline work?"
- "What is Pareto Merging?"
- "What prompt optimizers are available?"

**Configuration:**
- "What models does ScaleDown support?"
- "How do I set my API key?"
- "What exceptions does the SDK throw?"

## Deploying on AWS App Runner

App Runner is the recommended way to keep this bot running 24/7. It deploys directly from your GitHub repo — no Docker, no containers, no build tools on your machine.

### How it works

App Runner expects an HTTP service. The bot runs a lightweight health-check HTTP server on port 8080 (via [healthcheck.py](healthcheck.py)) alongside the Discord WebSocket connection. App Runner pings `/health`, gets a 200, and keeps the service alive.

The [apprunner.yaml](../apprunner.yaml) at the repo root tells App Runner how to build and run the bot. App Runner reads this file automatically.

### Prerequisites

- An AWS account
- This repo pushed to GitHub

That's it. No Docker, no CLI tools, no local builds.

### Step 1: Push this repo to GitHub

If you haven't already:

```bash
cd docs-main
git init
git add -A
git commit -m "ScaleDown docs + Discord bot"
git remote add origin https://github.com/YOUR_ORG/scaledown-docs.git
git push -u origin main
```

### Step 2: Connect GitHub to App Runner

1. Go to the **AWS Console** → search **App Runner** → **Create service**
2. Source: **Source code repository**
3. Click **Add new** to connect your GitHub account (one-time setup):
   - AWS will open a GitHub authorization flow
   - Select the repository: `YOUR_ORG/scaledown-docs`
   - Branch: `main`
4. Deployment trigger: choose **Manual** or **Automatic**
   - **Automatic** = App Runner redeploys every time you push to `main` (great for keeping docs fresh)
   - **Manual** = you trigger deploys yourself from the console

### Step 3: Configure build settings

1. Configuration source: **Configuration file** (App Runner will find the `apprunner.yaml` at the repo root)
2. Click **Next**

### Step 4: Configure the service

1. Service name: `scaledown-discord-bot`
2. **Environment variables** — click **Add environment variable** for each:

   | Key | Value |
   |---|---|
   | `DISCORD_BOT_TOKEN` | Your Discord bot token |
   | `SCALEDOWN_API_KEY` | Your ScaleDown API key |
   | `OPENAI_API_KEY` | Your OpenAI API key |
   | `LLM_MODEL` | `gpt-4o` |
   | `BOT_CHANNEL_NAME` | `scaledown-support` (or leave empty for all channels) |

   (`DOCS_PATH` and `PORT` are already set in `apprunner.yaml` — no need to add them here)

3. Instance size:
   - **CPU**: 0.25 vCPU
   - **Memory**: 0.5 GB
4. Health check:
   - Protocol: **HTTP**
   - Path: `/health`
   - Interval: `10` seconds
5. Click **Create & deploy**

### Step 5: Wait for deployment

App Runner will:
1. Pull your repo from GitHub
2. Run `pip install -r discord-bot/requirements.txt`
3. Start `python discord-bot/bot.py`
4. Begin health-checking `/health` on port 8080

This takes 2-3 minutes. You'll see the status go from **Creating** → **Running**.

### Verify it's working

Once the status shows **Running**:

1. Click the **Default domain** URL in the App Runner console (looks like `https://xxxxx.us-east-1.awsapprunner.com`)
2. Add `/health` to the URL — you should see `{"status": "healthy"}`
3. Add `/status` for details — shows bot connection state, guild count, doc chunks loaded
4. Go to your Discord server — the bot should be online and responding to `/ask`

### Updating when docs change

**If you chose Automatic deployment:** just `git push` — App Runner redeploys automatically.

**If you chose Manual deployment:**
1. Push your changes to GitHub
2. Go to App Runner console → your service → click **Deploy**

### Cost

| Resource | Cost |
|---|---|
| App Runner (0.25 vCPU, 0.5 GB, always running) | ~$5/month |
| ScaleDown API calls | Per your plan |
| OpenAI API calls | Per your plan |

---

## Health Check Endpoints

The bot exposes two HTTP endpoints (default port 8080):

| Endpoint | Description |
|---|---|
| `GET /health` | Simple health check — returns `{"status": "healthy"}` |
| `GET /status` | Detailed status — bot connection, guild count, QA engine state, uptime |

## Troubleshooting

| Problem | Solution |
|---|---|
| Bot doesn't respond | Check that **Message Content Intent** is enabled in Discord Developer Portal |
| Slash commands not showing | Wait up to 1 hour for Discord to propagate, or kick and re-invite the bot |
| `AuthenticationError` from ScaleDown | Verify your `SCALEDOWN_API_KEY` is valid and not expired |
| `FileNotFoundError` on startup | Check that `DOCS_PATH` points to the directory containing `.mdx`/`.md` files |
| Slow first response | First query compresses all doc chunks — subsequent queries are similar speed |
| Bot responds with error | Check **App Runner → Logs** (or CloudWatch) for details |
| `RuntimeError: DISCORD_BOT_TOKEN is not set` | Make sure env vars are set in App Runner service configuration |
| App Runner health check fails | Check that port 8080 is set and `/health` returns 200 |
| App Runner keeps restarting | Check CloudWatch logs — usually a missing or invalid env var |
| Build fails in App Runner | Check the build logs — usually a dependency issue in `requirements.txt` |
