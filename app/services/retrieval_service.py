from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class RetrievalService:

    def __init__(self):

        self.embedding_service = EmbeddingService()

        self.vector_store = None

    def add_documents(
        self,
        chunks: list[str]
    ):

        embeddings = (
            self.embedding_service.generate_embeddings(
                chunks
            )
        )

        dimension = embeddings.shape[1]

        if self.vector_store is None:

            self.vector_store = VectorStore(
                dimension
            )

        self.vector_store.add(
            embeddings,
            chunks
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        if self.vector_store is None:

            return []

        query_embedding = (
            self.embedding_service.generate_embeddings(
                [query]
            )[0]
        )

        return self.vector_store.search(
            query_embedding,
            top_k
        )