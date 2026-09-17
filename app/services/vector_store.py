import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:

    def __init__(
        self, dimension: int | None = None, storage_path: str = "data/vector_store"
    ):

        self.storage_path = Path(storage_path)

        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_path / "index.faiss"

        self.metadata_path = self.storage_path / "metadata.json"

        self.index = None
        self.documents = []

        self.load()

        if self.index is None and dimension is not None:

            self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings, documents: list[dict]):

        embeddings = np.asarray(embeddings, dtype="float32")

        if self.index is None:

            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.documents.extend(documents)

        self.save()

    def search(self, query_embedding, top_k: int = 5, score_threshold: float = 0.30):

        if self.index is None:
            return []

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray([query_embedding], dtype="float32")

        actual_top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, actual_top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            if score < score_threshold:
                continue

            document = self.documents[index]

            results.append(
                {
                    "text": document["text"],
                    "metadata": document["metadata"],
                    "score": float(score),
                }
            )

        return results

    def save(self):

        if self.index is not None:

            faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w", encoding="utf-8") as file:

            json.dump(self.documents, file, ensure_ascii=False, indent=2)

    def load(self):

        if self.index_path.exists():

            self.index = faiss.read_index(str(self.index_path))

        if self.metadata_path.exists():

            with open(self.metadata_path, "r", encoding="utf-8") as file:

                self.documents = json.load(file)

    def clear(self):

        self.index = None
        self.documents = []

        if self.index_path.exists():
            self.index_path.unlink()

        if self.metadata_path.exists():
            self.metadata_path.unlink()
