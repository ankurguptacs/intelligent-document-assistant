from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService


class RAGService:

    def __init__(self):

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore()

        self.llm_service = LLMService()

    def add_documents(self, documents: list[dict]):

        if not documents:
            return

        texts = [document["text"] for document in documents]

        embeddings = self.embedding_service.generate_embeddings(texts)

        self.vector_store.add(embeddings, documents)

    def search(self, query: str, top_k: int = 5):

        if self.vector_store.index is None:

            return []

        query_embedding = self.embedding_service.generate_embeddings([query])[0]

        return self.vector_store.search(
            query_embedding=query_embedding, top_k=top_k, score_threshold=0.30
        )

    def ask(self, question: str, top_k: int = 5):

        results = self.search(query=question, top_k=top_k)

        if not results:

            return {
                "answer": (
                    "I could not find sufficient "
                    "information in the uploaded "
                    "documents to answer this question."
                ),
                "sources": [],
            }

        context_parts = []

        for result in results:

            metadata = result["metadata"]

            filename = metadata.get("filename", "Unknown")

            page = metadata.get("page", "Unknown")

            context_parts.append(f"""
        SOURCE:
        Document: {filename}
        Page: {page}

        CONTENT:
        {result["text"]}
        """)

        context = "\n\n".join(context_parts)

        answer = self.llm_service.generate_answer(question=question, context=context)

        return {"answer": answer, "sources": results}

    def rebuild(self, documents: list[dict]):

        self.vector_store.clear()

        if not documents:
            return

        texts = [document["text"] for document in documents]

        embeddings = self.embedding_service.generate_embeddings(texts)

        self.vector_store.add(embeddings, documents)
