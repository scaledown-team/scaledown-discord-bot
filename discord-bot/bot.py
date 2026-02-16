#!/usr/bin/env python3
"""
ScaleDown Documentation Bot for Discord.

A Discord bot that answers questions about ScaleDown using the ScaleDown
compression REST API for query-aware context retrieval + compression,
then sends the compressed context to an LLM for answer generation.

Usage:
    1. Copy .env.example to .env and fill in your keys
    2. pip install -r requirements.txt
    3. python bot.py

Commands:
    /ask <question>      — Ask anything about ScaleDown
    /quickstart          — Get the ScaleDown quickstart guide
    /examples            — See common usage examples
    !ask <question>      — Prefix command alternative
    !quickstart          — Prefix quickstart guide
    !ping                — Check if the bot is alive
    !help_sd             — Show all commands
"""

import asyncio
import os
import textwrap
import threading

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from docs_loader import load_docs
from healthcheck import start_health_server
from qa_engine import QAEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DOCS_PATH = os.getenv("DOCS_PATH", "../")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
BOT_CHANNEL_NAME = os.getenv("BOT_CHANNEL_NAME", "")

if not DISCORD_TOKEN:
    raise RuntimeError(
        "DISCORD_BOT_TOKEN is not set. "
        "Copy .env.example to .env and add your bot token."
    )

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Will be initialized in on_ready
qa_engine: QAEngine | None = None

# ---------------------------------------------------------------------------
# Static content
# ---------------------------------------------------------------------------
QUICKSTART_TEXT = textwrap.dedent("""\
    **ScaleDown Quickstart Guide**

    ScaleDown compresses your AI prompts by 40-70% while preserving meaning. \
    Here's how to get started:

    **1. Get an API Key**
    Contact the ScaleDown sales team: <https://blog.scaledown.ai/blog/getting-started>

    **2. Make Your First API Call**
    ```python
    import requests

    url = "https://api.scaledown.xyz/compress/raw/"
    headers = {
        "x-api-key": "YOUR_API_KEY",
        "Content-Type": "application/json"
    }
    payload = {
        "context": "Your long context here...",
        "prompt": "Your question here",
        "model": "gpt-4o",
        "scaledown": {"rate": "auto"}
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    print(result["compressed_prompt"])
    print(f"Tokens: {result['original_prompt_tokens']} → {result['compressed_prompt_tokens']}")
    ```

    **3. Use With Your LLM**
    Send the `compressed_prompt` as context to OpenAI, Gemini, Claude, or any LLM.

    **Or use the Python SDK:**
    ```
    pip install scaledown
    ```
    ```python
    from scaledown.compressor import ScaleDownCompressor

    compressor = ScaleDownCompressor(target_model="gpt-4o", rate="auto")
    result = compressor.compress(context="...", prompt="...")
    print(result.content)
    result.print_stats()
    ```

    Ask me anything with `/ask` — I know the full ScaleDown docs!
""")

EXAMPLES_TEXT = textwrap.dedent("""\
    **ScaleDown Usage Examples**

    **Basic Compression (REST API):**
    ```python
    response = requests.post(
        "https://api.scaledown.xyz/compress/raw/",
        headers={"x-api-key": "KEY", "Content-Type": "application/json"},
        json={
            "context": "Long document...",
            "prompt": "What is the main argument?",
            "model": "gpt-4o",
            "scaledown": {"rate": "auto"}
        }
    )
    compressed = response.json()["compressed_prompt"]
    ```

    **Python SDK — Compression with keyword preservation:**
    ```python
    from scaledown.compressor import ScaleDownCompressor
    compressor = ScaleDownCompressor(
        target_model="gpt-4o",
        rate="auto",
        preserve_keywords=True,
        preserve_words=["SLA", "uptime"]
    )
    result = compressor.compress(context=doc_text, prompt=question)
    ```

    **Code Optimization with HASTE:**
    ```python
    from scaledown.optimizer import HasteOptimizer
    optimizer = HasteOptimizer(top_k=5, bfs_depth=2)
    result = optimizer.optimize(
        context=source_code,
        query="authentication logic",
        file_path="src/auth.py"
    )
    ```

    **RAG Without a Vector Database:**
    ```python
    # ScaleDown's compression is query-aware — just send
    # your docs as context and the question as prompt.
    # No embeddings, no vector store needed!
    compressed = requests.post(url, headers=headers, json={
        "context": knowledge_base,
        "prompt": user_question,
        "model": "gpt-4o",
        "scaledown": {"rate": "auto"}
    }).json()["compressed_prompt"]
    ```

    **Full Pipeline (SDK):**
    ```python
    from scaledown.pipeline import make_pipeline
    from scaledown.optimizer import SemanticOptimizer
    from scaledown.compressor import ScaleDownCompressor

    pipe = make_pipeline(
        ("retriever", SemanticOptimizer(top_k=3)),
        ("compressor", ScaleDownCompressor(rate="auto"))
    )
    result = pipe.run(context=docs, query=q, prompt=q, file_path="docs.txt")
    print(f"Savings: {result.savings_percent}%")
    ```

    Ask me for more details on any of these with `/ask`!
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MAX_DISCORD_LENGTH = 2000


def split_response(text: str) -> list[str]:
    """Split a long response into chunks that fit Discord's 2000-char limit."""
    if len(text) <= MAX_DISCORD_LENGTH:
        return [text]
    chunks = []
    while text:
        if len(text) <= MAX_DISCORD_LENGTH:
            chunks.append(text)
            break
        # Try to split at a newline near the limit
        split_at = text.rfind("\n", 0, MAX_DISCORD_LENGTH)
        if split_at == -1:
            split_at = MAX_DISCORD_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


async def process_question(question: str) -> str:
    """Run the QA engine and return the answer or an error message."""
    if not qa_engine:
        return (
            "I'm still loading the ScaleDown documentation. "
            "Please try again in a moment!"
        )
    try:
        return qa_engine.answer(question)
    except Exception as e:
        print(f"[bot] Error answering question: {e}")
        return (
            "Sorry, I encountered an error while processing your question. "
            "Please try again later or rephrase your question.\n\n"
            f"Error details: `{type(e).__name__}`"
        )


def is_allowed_channel(channel: discord.abc.Messageable) -> bool:
    """Check if the bot should respond in this channel."""
    if not BOT_CHANNEL_NAME:
        return True
    return getattr(channel, "name", "") == BOT_CHANNEL_NAME


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    global qa_engine
    print(f"[bot] Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"[bot] Loading documentation from: {DOCS_PATH}")

    knowledge_base = load_docs(DOCS_PATH)
    qa_engine = QAEngine(knowledge_base=knowledge_base, llm_model=LLM_MODEL)

    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        print(f"[bot] Synced {len(synced)} slash command(s)")
    except Exception as e:
        print(f"[bot] Failed to sync slash commands: {e}")

    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="/ask about ScaleDown",
        )
    )
    print("[bot] Ready!")


# ---------------------------------------------------------------------------
# Slash command: /ask
# ---------------------------------------------------------------------------
@bot.tree.command(
    name="ask",
    description="Ask any question about ScaleDown — quickstart, API, SDK, RAG, compression, and more",
)
@app_commands.describe(question="Your question about ScaleDown")
async def slash_ask(interaction: discord.Interaction, question: str):
    if not is_allowed_channel(interaction.channel):
        await interaction.response.send_message(
            f"Please use the #{BOT_CHANNEL_NAME} channel for questions.",
            ephemeral=True,
        )
        return

    await interaction.response.defer(thinking=True)
    answer = await process_question(question)

    header = f"**Q:** {question}\n\n"
    full_response = header + answer

    chunks = split_response(full_response)
    await interaction.followup.send(chunks[0])
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


# ---------------------------------------------------------------------------
# Slash command: /quickstart
# ---------------------------------------------------------------------------
@bot.tree.command(
    name="quickstart",
    description="Get the ScaleDown quickstart guide — first API call in minutes",
)
async def slash_quickstart(interaction: discord.Interaction):
    if not is_allowed_channel(interaction.channel):
        await interaction.response.send_message(
            f"Please use the #{BOT_CHANNEL_NAME} channel.",
            ephemeral=True,
        )
        return

    chunks = split_response(QUICKSTART_TEXT)
    await interaction.response.send_message(chunks[0])
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


# ---------------------------------------------------------------------------
# Slash command: /examples
# ---------------------------------------------------------------------------
@bot.tree.command(
    name="examples",
    description="See common ScaleDown usage examples — REST API, SDK, RAG, code compression",
)
async def slash_examples(interaction: discord.Interaction):
    if not is_allowed_channel(interaction.channel):
        await interaction.response.send_message(
            f"Please use the #{BOT_CHANNEL_NAME} channel.",
            ephemeral=True,
        )
        return

    chunks = split_response(EXAMPLES_TEXT)
    await interaction.response.send_message(chunks[0])
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


# ---------------------------------------------------------------------------
# Prefix commands
# ---------------------------------------------------------------------------
@bot.command(name="ask")
async def prefix_ask(ctx: commands.Context, *, question: str):
    """Ask any question about ScaleDown."""
    if not is_allowed_channel(ctx.channel):
        return
    async with ctx.typing():
        answer = await process_question(question)
    header = f"**Q:** {question}\n\n"
    for chunk in split_response(header + answer):
        await ctx.reply(chunk)


@bot.command(name="quickstart")
async def prefix_quickstart(ctx: commands.Context):
    """Get the ScaleDown quickstart guide."""
    if not is_allowed_channel(ctx.channel):
        return
    for chunk in split_response(QUICKSTART_TEXT):
        await ctx.reply(chunk)


@bot.command(name="examples")
async def prefix_examples(ctx: commands.Context):
    """See common ScaleDown usage examples."""
    if not is_allowed_channel(ctx.channel):
        return
    for chunk in split_response(EXAMPLES_TEXT):
        await ctx.reply(chunk)


@bot.command(name="ping")
async def ping(ctx: commands.Context):
    """Check if the bot is alive."""
    latency_ms = round(bot.latency * 1000)
    docs_status = "loaded" if qa_engine else "loading..."
    await ctx.reply(f"Pong! Latency: {latency_ms}ms | Docs: {docs_status}")


@bot.command(name="help_sd")
async def help_sd(ctx: commands.Context):
    """Show help for the ScaleDown bot."""
    help_text = textwrap.dedent("""\
        **ScaleDown Documentation Bot**

        I'm your expert guide to the ScaleDown context engineering platform. \
        I know the full documentation — quickstart, REST API, Python SDK, \
        compression techniques, RAG pipelines, code optimization, \
        hallucination reduction, and more.

        **Commands:**
        `/ask <question>` — Ask anything about ScaleDown
        `/quickstart` — Get the quickstart guide with your first API call
        `/examples` — See common code examples
        `!ask <question>` — Prefix alternative for /ask
        `!quickstart` — Prefix alternative for /quickstart
        `!examples` — Prefix alternative for /examples
        `!ping` — Check if I'm online
        `!help_sd` — This message

        **What can I help with?**
        - Getting started & first API call
        - Installation (`pip install scaledown`)
        - REST API usage (any language)
        - Python SDK classes: `ScaleDownCompressor`, `HasteOptimizer`, \
    `SemanticOptimizer`, `Pipeline`
        - Building RAG without a vector database
        - Code compression with AST-based HASTE
        - Hallucination reduction pipeline
        - Pareto Merging for reasoning optimization
        - Prompt anatomy & best practices
        - Supported models & configuration
        - Error handling & troubleshooting

        **Example questions:**
        - "How do I get started with ScaleDown?"
        - "Show me how to compress a prompt with the REST API"
        - "What parameters does ScaleDownCompressor accept?"
        - "How do I build RAG without a vector database?"
        - "What is the HasteOptimizer and when should I use it?"
        - "How does the hallucination reduction pipeline work?"
        - "What's the difference between rate=auto and a fixed rate?"
        - "How do I preserve specific keywords during compression?"

        Just ask — I'm here to help!
    """)
    for chunk in split_response(help_text):
        await ctx.reply(chunk)


# ---------------------------------------------------------------------------
# Auto-reply when bot is mentioned
# ---------------------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    # If the bot is mentioned, treat the message as a question
    if bot.user in message.mentions:
        content = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if content and is_allowed_channel(message.channel):
            async with message.channel.typing():
                answer = await process_question(content)
            for chunk in split_response(answer):
                await message.reply(chunk)
            return

    # Process prefix commands
    await bot.process_commands(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Start the HTTP health-check server in a background thread.
    # AWS App Runner requires a listening HTTP port for health checks.
    health_port = int(os.getenv("PORT", "8080"))
    start_health_server(health_port, bot_ref=bot, qa_ref=lambda: qa_engine)

    bot.run(DISCORD_TOKEN)
