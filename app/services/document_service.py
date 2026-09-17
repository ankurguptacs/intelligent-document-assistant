import fitz


class DocumentService:

    def extract_pages(self, file_path: str) -> list[dict]:
        """
        Extract text page-by-page from a PDF.
        """

        document = fitz.open(file_path)

        pages = []

        for page_number, page in enumerate(document):

            text = page.get_text().strip()

            if text:

                pages.append({
                    "page": page_number + 1,
                    "text": text
                })

        document.close()

        return pages


    def create_chunks(
        self,
        pages: list[dict],
        filename: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> list[dict]:
        """
        Create chunks while preserving page and document metadata.
        """

        chunks = []

        chunk_id = 0

        for page in pages:

            text = page["text"]

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunk_text = text[start:end].strip()

                if chunk_text:

                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "filename": filename,
                            "page": page["page"],
                            "chunk_id": chunk_id
                        }
                    })

                    chunk_id += 1

                start += chunk_size - overlap

        return chunks