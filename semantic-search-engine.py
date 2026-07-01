from sentence_transformers import SentenceTransformer, util
import time
import hashlib
from typing import List, Dict


class SemanticSearchEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the search engine with a specific embedding model.

        Args:
            model_name: The name of the SentenceTransformer model to use
        """
        # TODO: Initialize the model and necessary data structures
        # - Initialize the embedding model
        # - Create empty data structures for document storage
        # - Initialize a cache for query embeddings
        pass

    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 32) -> None:
        """
        Process and add documents to the search engine.

        Args:
            documents: List of document dictionaries with 'id' and 'content' keys
            batch_size: Batch size for efficient embedding generation
        """
        # TODO: Process documents in batches and store their embeddings
        # - Process documents in batches of the specified size
        # - Generate embeddings for each batch
        # - Store documents with their embeddings in your data structure
        pass

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text, using cache if available.

        Args:
            text: The text to embed

        Returns:
            The embedding vector for the text as a list
        """
        # TODO: Implement caching for embeddings
        # - Create a hash of the input text to use as a cache key
        # - Check if the embedding exists in the cache
        # - If it does, increment cache hit counter and return cached embedding
        # - If not, generate the embedding, cache it, increment cache miss counter
        # - Return the embedding
        pass

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for documents most similar to the query.

        Args:
            query: The search query text
            top_k: Number of top results to return

        Returns:
            List of top_k documents with their similarity scores
        """
        # TODO: Implement semantic search functionality
        # - Get embedding for the query (using cache if possible)
        # - Calculate similarity between query and all documents
        # - Sort the results by similarity score in descending order
        # - Return top_k results with their scores and document data
        pass

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Return statistics about the cache performance.

        Returns:
            Dictionary with cache hit/miss statistics
        """
        # TODO: Return cache statistics
        # - Calculate total cache accesses
        # - Calculate hit rate percentage
        # - Return a dictionary with hits, misses, total, and hit rate
        pass


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        {"id": "doc1", "content": "How to reset your password in our application"},
        {"id": "doc2", "content": "Troubleshooting login issues and account access problems"},
        {"id": "doc3", "content": "Understanding your monthly billing statement"},
        {"id": "doc4", "content": "How to upgrade your subscription plan"},
        {"id": "doc5", "content": "Setting up two-factor authentication for security"},
    ]

    # Sample queries
    queries = [
        "I forgot my password",
        "Can't log into my account",
        "How do I understand my bill",
        "I want to upgrade my account",
        "password reset",
        "I forgot my password",  # Repeated query to test caching
    ]

    # Initialize and use the search engine
    search_engine = SemanticSearchEngine()

    # Add documents
    start_time = time.time()
    search_engine.add_documents(documents)
    print(f"Document processing time: {time.time() - start_time:.4f}s")

    # Search with each query
    for query in queries:
        start_time = time.time()
        results = search_engine.search(query)
        print(f"\nQuery: '{query}'")
        print(f"Search time: {time.time() - start_time:.4f}s")

        for result in results:
            print(
                f"  - {result['id']} (Score: {result['score']:.4f}): {result['content']}")

    # Print cache statistics
    print("\nCache statistics:")
    stats = search_engine.get_cache_stats()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Total cache accesses: {stats['total']}")
    print(f"Hit rate: {stats['hit_rate_percent']:.2f}%")