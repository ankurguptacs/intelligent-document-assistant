from fastapi import APIRouter
from pydantic import BaseModel

from app.core.dependencies import rag_service


router = APIRouter()


class ChatRequest(BaseModel):

    question: str

    top_k: int = 5


@router.post("/")
async def chat(
    request: ChatRequest
):

    result = rag_service.ask(
        question=request.question,
        top_k=request.top_k
    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }