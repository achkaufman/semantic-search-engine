from sentence_transformers import SentenceTransformer, util
import hashlib
import json
from pathlib import Path
from typing import List, Dict


class SemanticSearchEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the search engine with a specific embedding model.

        Args:
            model_name: The name of the SentenceTransformer model to use
        """
        # - Initialize the embedding model
        self.model = SentenceTransformer(model_name)

        # - Create empty data structures for document storage
        self.documents = []

        # - Initialize a cache for query embeddings
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0


    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 32) -> None:
        """
        Process and add documents to the search engine.

        Args:
            documents: List of document dictionaries with 'id', 'title', and
                'content' keys. Documents are embedded and searched by 'title'.
            batch_size: Batch size for efficient embedding generation
        """
        # - Process documents in batches of the specified size
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # - Generate embeddings for the current batch (based on the title)
            titles = [doc['title'] for doc in batch]
            embeddings = self.model.encode(titles)

            # - Store documents with their embeddings in your data structure
            for doc, embedding in zip(batch, embeddings):
                self.documents.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'content': doc['content'],
                    'embedding': embedding
                })


    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text, using cache if available.

        Args:
            text: The text to embed

        Returns:
            The embedding vector for the text as a list
        """
        # - Create a hash of the input text to use as a cache key
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()

        # - Check if the embedding exists in the cache
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]

        # - If not, generate the embedding, cache it, increment cache miss counter
        embedding = self.model.encode(text)
        self.embedding_cache[cache_key] = embedding
        self.cache_misses += 1

        # - Return the embedding
        return embedding
    

    def retrieve(self, query: str, top_n: int = 20) -> List[Dict[str, any]]:
        """
        Retrieve an initial candidate set of documents for a query.

        This is the bi-encoder retrieval stage: it encodes the query, scores it
        against every document by cosine similarity, and returns the top_n
        highest-scoring candidates. A later stage (e.g. a reranker) can reorder
        this candidate set before the final results are shown.

        Args:
            query: The search query text
            top_n: Number of candidate documents to retrieve

        Returns:
            List of up to top_n candidate documents with their similarity
            scores, ordered from most to least similar
        """
        # - Get embedding for the query (using cache if possible)
        query_embedding = self._get_embedding(query)

        # - Calculate similarity between query and all documents
        results = []
        for doc in self.documents:
            score = util.cos_sim(query_embedding, doc['embedding']).item()
            results.append({
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
                'score': float(score)
            })

        # - Sort the results by similarity score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)

        # - Return the top_n candidates
        return results[:top_n]

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Return statistics about the cache performance.

        Returns:
            Dictionary with cache hit/miss statistics
        """
        # - Calculate total cache accesses
        total_accesses = self.cache_hits + self.cache_misses

        # - Calculate hit rate percentage
        hit_rate = (self.cache_hits / total_accesses) * 100 if total_accesses > 0 else 0.0

        # - Return a dictionary with hits, misses, total, and hit rate
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total_accesses,
            'hit_rate_percent': hit_rate,
        }


def load_documents(docs_dir: str = "docs") -> List[Dict[str, str]]:
    """
    Load documents from a folder of JSON files.

    Each JSON file is expected to contain an object with 'id' and 'content'
    keys, matching the schema the search engine indexes.

    Args:
        docs_dir: Path to the folder containing the document JSON files.

    Returns:
        A list of document dictionaries, sorted by file name for stable order.
    """
    documents = []
    for path in sorted(Path(docs_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            documents.append(json.load(f))
    return documents


# Command line interface
if __name__ == "__main__":
    # Load the documents to search over from the docs/ folder
    documents = load_documents()

    # Build the search engine and index the documents
    search_engine = SemanticSearchEngine()
    search_engine.add_documents(documents)

    # Retrieve a larger candidate set from the bi-encoder, but only show the
    # top few to the user. The extra candidates leave room for a future
    # reranking stage.
    CANDIDATE_COUNT = 5
    TOP_K_DISPLAY = 3

    # Interactive prompt: keep asking for queries until the user exits.
    print("Semantic search. Type a query and press Enter.")
    print("Press Enter on an empty line or Ctrl-C to quit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # An empty query exits the loop.
        if not query:
            print("Goodbye!")
            break

        # Retrieve a candidate set by title, then show only the top matches.
        candidates = search_engine.retrieve(query, top_n=CANDIDATE_COUNT)
        results = candidates[:TOP_K_DISPLAY]

        if not results:
            print("\nNo matching documents found.\n")
            continue

        print(f"\nTop {len(results)} results for: '{query}'\n")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result['title']} (score: {result['score']:.4f})")
        print()

        # Let the user open one of the results to read its content.
        try:
            selection = input(
                "Select a result number to read it (or press Enter to search again)> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # An empty selection goes back to a new query.
        if not selection:
            print()
            continue

        # Validate the selection, then print the chosen document's content.
        if selection.isdigit() and 1 <= int(selection) <= len(results):
            chosen = results[int(selection) - 1]
            print(f"\n=== {chosen['title']} ===\n")
            print(chosen['content'])
            print()
        else:
            print("Invalid selection. Returning to search.\n")