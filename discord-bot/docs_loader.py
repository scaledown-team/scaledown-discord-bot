"""
Loads and preprocesses ScaleDown documentation files (.mdx, .md) into
a single knowledge base string for the QA engine.

Preserves document structure (headings, code blocks, tables) while
stripping MDX/JSX component wrappers and formatting noise.
"""

import os
import re
from pathlib import Path


# MDX/JSX component tags to strip (Mintlify components, HTML-like wrappers)
_TAG_PATTERN = re.compile(
    r"</?(?:Card|CardGroup|Steps|Step|Tabs|Tab|Tip|Note|Info|Warning|Check|"
    r"Accordion|AccordionGroup|Expandable|CodeGroup|ResponseField|ParamField|"
    r"Frame|Icon|Tooltip|Snippet|RequestExample|ResponseExample|"
    r"img|br|sub|sup|div|span)[^>]*>",
    re.IGNORECASE,
)

# Frontmatter block (--- ... ---)
_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

# Mermaid diagrams (```mermaid ... ```) — visual only, not useful as text
_MERMAID = re.compile(r"```mermaid.*?```", re.DOTALL)

# Import statements in MDX
_IMPORTS = re.compile(r"^import\s+.*$", re.MULTILINE)

# Image references ![alt](/path) — not useful in text context
_IMAGES = re.compile(r"!\[.*?\]\(.*?\)")

# Style attribute blocks (style={{...}})
_STYLE_ATTRS = re.compile(r"style=\{\{.*?\}\}", re.DOTALL)

# Consecutive blank lines → max 2
_BLANK_LINES = re.compile(r"\n{3,}")

# Map of filenames to human-readable topic names for better context
_TOPIC_NAMES = {
    "quickstart.mdx": "Getting Started & Quickstart Guide",
    "Documentation.mdx": "Python SDK & API Reference",
    "workflow_example.mdx": "Integration Workflow Examples",
    "rag_example.mdx": "Building RAG Without a Vector Database",
    "anatomy.mdx": "Prompt Anatomy & Best Practices",
    "development.mdx": "Development Workflow & Advanced Usage",
    "web-editor.mdx": "Web Editor",
    "README.md": "ScaleDown Overview",
    "pareto_merging.mdx": "Pareto Merging — Reasoning Model Optimization",
    "semantic_ast_compression.mdx": "Semantic AST Code Compression",
    "overview.mdx": "Hallucination Reduction Pipeline Overview",
    "modular-prompt-optimization.mdx": "Modular Prompt Optimization Framework",
    "hallucination-reductiono-pipeline.mdx": "Hallucination Reduction Pipeline Integration",
    "semantic-audio-tokenizer.mdx": "Semantic Audio Tokenizer",
}


def _clean_mdx(text: str, source_file: str) -> str:
    """Strip MDX component wrappers and formatting noise, keep prose + code."""
    text = _FRONTMATTER.sub("", text)
    text = _IMPORTS.sub("", text)
    text = _MERMAID.sub("", text)
    text = _IMAGES.sub("", text)
    text = _STYLE_ATTRS.sub("", text)
    text = _TAG_PATTERN.sub("", text)
    text = _BLANK_LINES.sub("\n\n", text)

    # Determine a descriptive topic name
    fname = Path(source_file).name
    topic = _TOPIC_NAMES.get(fname, source_file)

    heading = f"{'=' * 60}\nTOPIC: {topic}\nSource: {source_file}\n{'=' * 60}\n\n"
    return heading + text.strip()


def load_docs(docs_path: str) -> str:
    """
    Recursively load all .mdx and .md files under `docs_path`,
    clean them, and concatenate into a single knowledge base string.

    Returns:
        A single string with all documentation content separated by markers.
    """
    docs_path = Path(docs_path).resolve()
    sections: list[str] = []

    # Walk the tree, skip hidden dirs and non-doc directories
    for root, dirs, files in os.walk(docs_path):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in ("node_modules", "discord-bot", "pictures", "writing-content")
        ]
        for fname in sorted(files):
            if fname.endswith((".mdx", ".md")):
                fpath = Path(root) / fname
                rel = fpath.relative_to(docs_path)
                content = fpath.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    cleaned = _clean_mdx(content, str(rel))
                    sections.append(cleaned)

    if not sections:
        raise FileNotFoundError(
            f"No .mdx or .md files found under {docs_path}"
        )

    knowledge_base = "\n\n\n".join(sections)
    total_chars = len(knowledge_base)
    print(
        f"[docs_loader] Loaded {len(sections)} documents "
        f"({total_chars:,} chars, ~{total_chars // 4:,} tokens est.)"
    )
    return knowledge_base
