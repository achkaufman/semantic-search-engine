# Semantic Search Engine

A small, interactive semantic search engine for support documentation, built as
the exercise from the LinkedIn Learning *Fundamentals of AI Engineering* course.
It embeds document titles with [sentence-transformers](https://www.sbert.net/)
and ranks them against your query by meaning rather than keywords, then lets you
open a result to read its full content.

## Features

- Semantic (meaning-based) search over a folder of documents.
- Interactive command-line loop: search, pick a result, read it, repeat.
- Documents are loaded from JSON files in `docs/` — no code changes to add more.
- Query-embedding cache so repeated searches return instantly.

## Requirements

- Python 3.9+
- The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB).

## Installation

```bash
# Clone, then create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the app (make sure the virtual environment is activated):

```bash
python3 semantic-search-engine.py
```

You'll get an interactive prompt:

```
Semantic search. Type a query and press Enter.
Press Enter on an empty line or Ctrl-C to quit.

Query> how do I get my money back

Top 3 results for: 'how do I get my money back'

1. How to cancel your subscription and request a refund (score: 0.5000)
2. Contacting customer support and submitting a help ticket (score: 0.3438)
3. Updating your payment method and billing information (score: 0.3015)

Select a result number to read it (or press Enter to search again)> 1

=== How to cancel your subscription and request a refund ===

If our service no longer meets your needs, you can cancel your subscription...
```

- Type a query and press **Enter** to search.
- Enter a **result number** to read that document, or press **Enter** to search again.
- Press **Enter on an empty query** or **Ctrl-C** to quit.

## Documents

Documents live in the `docs/` folder, one JSON file per document. Each file uses
this schema:

```json
{
    "id": "doc1",
    "title": "How to reset your password in our application",
    "content": "Full documentation text shown when the result is selected..."
}
```

- `title` is what the engine embeds and searches against.
- `content` is the body displayed when you open a result.

To add a document, drop a new `*.json` file into `docs/` and restart the app.

## Project structure

```
semantic-search-engine.py   # search engine + interactive CLI
docs/                        # document JSON files (id, title, content)
requirements.txt             # Python dependencies
```

## How it works

On startup the app loads every document from `docs/`, loads the embedding model
once, and encodes each document's title into a vector. For each query it embeds
the query text (caching the result), computes cosine similarity against all
document titles, and returns the top 3 matches. Selecting a result prints that
document's stored content.

> This is an in-memory engine intended for learning and small document sets. For
> large-scale use, the linear similarity search would be replaced by a vector
> database (e.g. FAISS, Qdrant, or pgvector).
