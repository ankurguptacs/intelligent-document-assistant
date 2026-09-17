import json
from pathlib import Path


class DocumentRegistry:

    def __init__(self, storage_path: str = "data/documents/registry.json"):

        self.storage_path = Path(storage_path)

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.documents = {}

        self.load()

    def load(self):

        if not self.storage_path.exists():
            return

        with open(self.storage_path, "r", encoding="utf-8") as file:

            self.documents = json.load(file)

    def save(self):

        with open(self.storage_path, "w", encoding="utf-8") as file:

            json.dump(self.documents, file, ensure_ascii=False, indent=2)

    def exists(self, filename: str) -> bool:

        return filename in self.documents

    def add(self, filename: str, page_count: int, chunk_count: int):

        self.documents[filename] = {
            "filename": filename,
            "page_count": page_count,
            "chunk_count": chunk_count,
        }

        self.save()

    def get_all(self):

        return list(self.documents.values())

    def remove(self, filename: str) -> bool:

        if filename not in self.documents:
            return False

        del self.documents[filename]

        self.save()

        return True
