import requests
import os


class LLMService:

    def __init__(
        self,
        model: str = os.getenv("LLM_MODEL", "gemma3"),
        ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434"),
    ):

        self.model = model
        self.ollama_url = ollama_url

    def generate_answer(self, question: str, context: str):

        prompt = f"""
You are an intelligent document question-answering assistant.

Your job is to answer the user's question using ONLY the
information provided in the CONTEXT below.

Rules:

1. Use only the provided context.
2. Do not use your own general knowledge.
3. Do not invent or assume information.
4. If the answer cannot be found in the context, say:
   "I could not find this information in the uploaded documents."
5. Give a clear and concise answer.
6. If multiple pieces of context are relevant, combine them logically.

CONTEXT:
--------------------
{context}
--------------------

QUESTION:
{question}

ANSWER:
"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "No answer was generated.")
