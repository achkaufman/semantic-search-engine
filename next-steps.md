# Next Steps

A roadmap of improvements for the semantic search engine, ordered from lowest to
highest priority/complexity. Earlier items are quick, high-value foundations;
later items are larger investments that unlock scale, integrations, and new
capabilities.

---

## 1. Add automated tests

**Description:** Lock in current behavior with a test suite so future changes
don't silently break search results or the CLI flow.

**High-level steps:**
- Add `pytest` and stub the embedding model with a deterministic fake encoder.
- Unit-test `load_documents`, `add_documents`, `search`, and the cache stats.
- Wire the tests into a CI workflow (e.g. GitHub Actions).

---

## 2. Externalize configuration and CLI options

**Description:** Move hard-coded values (model name, `docs/` path, `top_k`) into
configuration and command-line flags so behavior can change without editing code.

**High-level steps:**
- Add `argparse` flags such as `--docs`, `--model`, and `--top-k`.
- Support environment variables / a `.env` file for defaults.
- Document the options in the README.

---

## 3. Persist embeddings between runs

**Description:** Today every startup re-embeds all documents. Caching embeddings
to disk makes startup fast and only re-embeds new or changed documents.

**High-level steps:**
- Hash each document's title/content to detect changes.
- Save embeddings (e.g. `numpy`/Parquet) alongside the doc id.
- On startup, load cached vectors and embed only new/modified docs.

---

## 4. Improve relevance with full-content embeddings and chunking

**Description:** Searching only on titles misses relevant body text. Embedding
the content (split into passages for long docs) improves recall and precision.

**High-level steps:**
- Split long `content` into overlapping chunks with a `parent_id`.
- Embed titles and chunks; map chunk hits back to their source document.
- Tune how title vs. content scores are combined.

---

## 5. Add hybrid search and re-ranking

**Description:** Combine keyword (lexical) search with semantic search, then
re-rank the top candidates with a stronger model for better accuracy on
edge-case queries.

**High-level steps:**
- Add a lexical index (e.g. BM25 via `rank-bm25`).
- Merge lexical and semantic scores into a single ranking.
- Optionally re-rank the top-N with a cross-encoder model.

---

## 6. Replace the linear scan with a vector database

**Description:** The current search compares the query against every document in
a loop, which won't scale. A vector store provides fast approximate
nearest-neighbor search over millions of documents.

**High-level steps:**
- Choose a store (FAISS in-process, or Qdrant/Chroma/pgvector as a service).
- Index document embeddings with their ids/metadata.
- Replace the manual `cos_sim` loop with the store's query API.

---

## 7. Expose the engine as a web API and containerize it

**Description:** Turn the CLI into a service other apps can call over HTTP, and
package it with Docker for consistent local and cloud deployment.

**High-level steps:**
- Add a FastAPI app with `/search` and `/documents` endpoints.
- Load the model once at startup and reuse it across requests.
- Write a `Dockerfile` (and optional `docker-compose.yml`) to run it anywhere.

---

## 8. Integrate external document sources

**Description:** Pull documentation from real systems (a CMS, a database, cloud
storage, or a knowledge base) instead of local JSON files, with automatic
re-indexing when content changes.

**High-level steps:**
- Add source connectors (e.g. database, S3, REST API, or Notion/Confluence).
- Normalize each source into the `id`/`title`/`content` schema.
- Schedule incremental ingestion so new/updated docs are re-embedded.

---

## 9. Add observability and search-quality evaluation

**Description:** Measure and improve real-world search quality with logging,
metrics, and an evaluation set, plus a feedback loop from users.

**High-level steps:**
- Log queries, latency, and selected results (privacy-permitting).
- Build a labeled query set and track metrics like recall@k / MRR.
- Capture user feedback ("was this helpful?") to guide tuning.
