from fastapi import FastAPI

from app.api.document import router as documents_router
from app.api.chat import router as chat_router
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = FastAPI(
    title="Intelligent Document Q&A Assistant",
    description="AI-powered document question-answering system using RAG and LLMs",
    version="1.0.0",
)

app.include_router(documents_router, prefix="/documents", tags=["Documents"])

app.include_router(chat_router, prefix="/chat", tags=["Chat"])


@app.get("/")
def root():
    return {"message": "Intelligent Document Q&A Assistant is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
