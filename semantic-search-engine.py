from sentence_transformers import SentenceTransformer, util
import argparse
import hashlib
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
            documents: List of document dictionaries with 'id' and 'content' keys
            batch_size: Batch size for efficient embedding generation
        """
        # - Process documents in batches of the specified size
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # - Generate embeddings for the current batch
            contents = [doc['content'] for doc in batch]
            embeddings = self.model.encode(contents)

            # - Store documents with their embeddings in your data structure
            for doc, embedding in zip(batch, embeddings):
                self.documents.append({
                    'id': doc['id'],
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
    

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for documents most similar to the query.

        Args:
            query: The search query text
            top_k: Number of top results to return

        Returns:
            List of top_k documents with their similarity scores
        """
        # - Get embedding for the query (using cache if possible)
        query_embedding = self._get_embedding(query)

        # - Calculate similarity between query and all documents
        results = []
        for doc in self.documents:
            score = util.cos_sim(query_embedding, doc['embedding']).item()
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'score': float(score)
            })

        # - Sort the results by similarity score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)

        # - Return top_k results with their scores and document data
        return results[:top_k]

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


# Command line interface
if __name__ == "__main__":
    # Documents to search over
    documents = [
        {"id": "doc1", "content": "How to reset your password in our application"},
        {"id": "doc2", "content": "Troubleshooting login issues and account access problems"},
        {"id": "doc3", "content": "Understanding your monthly billing statement"},
        {"id": "doc4", "content": "How to upgrade your subscription plan"},
        {"id": "doc5", "content": "Setting up two-factor authentication for security"},
    ]

    # Parse the query from the command line
    parser = argparse.ArgumentParser(
        description="Search the documents for the ones most similar to your query."
    )
    parser.add_argument("query", help="The search query string.")
    args = parser.parse_args()

    # Build the search engine and index the documents
    search_engine = SemanticSearchEngine()
    search_engine.add_documents(documents)

    # Return the top 3 documents, ordered by similarity score
    results = search_engine.search(args.query, top_k=3)

    print(f"\nTop {len(results)} results for: '{args.query}'\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['id']} (score: {result['score']:.4f}): {result['content']}")