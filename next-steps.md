# Next Steps

A roadmap of improvements for the semantic search engine. The first three items
are retrieval-quality capabilities covered in the course, placed first so they
can be tackled one at a time as hands-on learning experiments (build order:
retrieve, then rerank, then fuse). The remaining items are ordered roughly from
lowest to highest priority/complexity — quick, high-value foundations first,
then larger investments that unlock scale, integrations, and new capabilities.

---

## 1. Implement a bi-encoder retriever

**Description:** Formalize the current dense search into an explicit bi-encoder
retriever that encodes queries and documents independently and returns an
initial candidate set by vector similarity. This is the foundation (stage one)
of a two-stage retrieve-then-rerank pipeline.

**High-level steps:**
- Extract retrieval into a `Retriever` component that embeds docs and queries
  with a bi-encoder (e.g. `all-MiniLM-L6-v2`).
- Return an initial top-N candidate set (N larger than the final top-k) to leave
  room for a downstream reranker.
- Keep document embeddings precomputed so initial retrieval stays fast.

---

## 2. Add a cross-encoder reranker

**Description:** After the bi-encoder retrieves candidates, use a cross-encoder
that jointly encodes each (query, document) pair to score relevance more
accurately, then reorder the candidates. This two-stage retrieve-then-rerank
approach sharpens precision at the top of the results.

**High-level steps:**
- Load a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) via
  `sentence-transformers`.
- Score each retrieved (query, document) pair and sort by the cross-encoder score.
- Retrieve top-N with the bi-encoder, rerank, and return the final top-k.

---

## 3. Add hybrid search with weighted fusion (BM25 + vector)

**Description:** Combine lexical BM25 retrieval with dense bi-encoder retrieval,
fusing their scores with a tunable weight so results benefit from both exact
keyword matches and semantic similarity. The cross-encoder reranker can then run
over the fused candidate set.

**High-level steps:**
- Add a BM25 index (e.g. `rank-bm25`) over the document text.
- Normalize and combine BM25 and vector scores via weighted fusion (or
  Reciprocal Rank Fusion).
- Expose the fusion weight as a config option and feed the fused candidates into
  the reranker.

---

## 4. Add automated tests

**Description:** Lock in current behavior with a test suite so future changes
don't silently break search results or the CLI flow.

**High-level steps:**
- Add `pytest` and stub the embedding model with a deterministic fake encoder.
- Unit-test `load_documents`, `add_documents`, `search`, and the cache stats.
- Wire the tests into a CI workflow (e.g. GitHub Actions).

---

## 5. Externalize configuration and CLI options

**Description:** Move hard-coded values (model name, `docs/` path, `top_k`) into
configuration and command-line flags so behavior can change without editing code.

**High-level steps:**
- Add `argparse` flags such as `--docs`, `--model`, and `--top-k`.
- Support environment variables / a `.env` file for defaults.
- Document the options in the README.

---

## 6. Persist embeddings between runs

**Description:** Today every startup re-embeds all documents. Caching embeddings
to disk makes startup fast and only re-embeds new or changed documents.

**High-level steps:**
- Hash each document's title/content to detect changes.
- Save embeddings (e.g. `numpy`/Parquet) alongside the doc id.
- On startup, load cached vectors and embed only new/modified docs.

---

## 7. Improve relevance with full-content embeddings and chunking

**Description:** Searching only on titles misses relevant body text. Embedding
the content (split into passages for long docs) improves recall and precision.

**High-level steps:**
- Split long `content` into overlapping chunks with a `parent_id`.
- Embed titles and chunks; map chunk hits back to their source document.
- Tune how title vs. content scores are combined.

---

## 8. Replace the linear scan with a vector database

**Description:** The current search compares the query against every document in
a loop, which won't scale. A vector store provides fast approximate
nearest-neighbor search over millions of documents.

**High-level steps:**
- Choose a store (FAISS in-process, or Qdrant/Chroma/pgvector as a service).
- Index document embeddings with their ids/metadata.
- Replace the manual `cos_sim` loop with the store's query API.

---

## 9. Expose the engine as a web API and containerize it

**Description:** Turn the CLI into a service other apps can call over HTTP, and
package it with Docker for consistent local and cloud deployment.

**High-level steps:**
- Add a FastAPI app with `/search` and `/documents` endpoints.
- Load the model once at startup and reuse it across requests.
- Write a `Dockerfile` (and optional `docker-compose.yml`) to run it anywhere.

---

## 10. Integrate external document sources

**Description:** Pull documentation from real systems (a CMS, a database, cloud
storage, or a knowledge base) instead of local JSON files, with automatic
re-indexing when content changes.

**High-level steps:**
- Add source connectors (e.g. database, S3, REST API, or Notion/Confluence).
- Normalize each source into the `id`/`title`/`content` schema.
- Schedule incremental ingestion so new/updated docs are re-embedded.

---

## 11. Add observability and search-quality evaluation

**Description:** Measure and improve real-world search quality with logging,
metrics, and an evaluation set, plus a feedback loop from users.

**High-level steps:**
- Log queries, latency, and selected results (privacy-permitting).
- Build a labeled query set and track metrics like recall@k / MRR.
- Capture user feedback ("was this helpful?") to guide tuning.
