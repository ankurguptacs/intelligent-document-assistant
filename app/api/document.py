from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil

from app.services.document_service import DocumentService
from app.core.dependencies import rag_service
from app.services.document_registry import DocumentRegistry

router = APIRouter()

document_service = DocumentService()
document_registry = DocumentRegistry()

UPLOAD_DIR = Path("data/documents")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:

        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if document_registry.exists(file.filename):

        raise HTTPException(
            status_code=409,
            detail=(f"Document '{file.filename}' " "is already indexed."),
        )

    file_path = UPLOAD_DIR / file.filename

    try:

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(file.file, buffer)

        pages = document_service.extract_pages(str(file_path))

        documents = document_service.create_chunks(pages=pages, filename=file.filename)

        rag_service.add_documents(documents)

        document_registry.add(
            filename=file.filename, page_count=len(pages), chunk_count=len(documents)
        )

        return {
            "filename": file.filename,
            "page_count": len(pages),
            "chunk_count": len(documents),
            "message": ("Document processed and indexed successfully"),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


class SearchRequest(BaseModel):

    query: str

    top_k: int = 5


@router.post("/search")
async def search_documents(request: SearchRequest):

    results = rag_service.search(query=request.query, top_k=request.top_k)

    return {"query": request.query, "results": results}


@router.get("/")
async def get_documents():

    return {"documents": document_registry.get_all()}


@router.post("/rebuild")
async def rebuild_vector_store():

    all_documents = []

    for document_info in document_registry.get_all():

        filename = document_info["filename"]

        file_path = UPLOAD_DIR / filename

        if not file_path.exists():
            continue

        pages = document_service.extract_pages(str(file_path))

        chunks = document_service.create_chunks(pages=pages, filename=filename)

        all_documents.extend(chunks)

    rag_service.rebuild(all_documents)

    return {
        "message": "Vector store rebuilt successfully",
        "document_count": len(document_registry.get_all()),
        "chunk_count": len(all_documents),
    }


@router.delete("/{filename}")
async def delete_document(filename: str):

    if not document_registry.exists(filename):

        raise HTTPException(status_code=404, detail="Document not found")

    file_path = UPLOAD_DIR / filename

    # Remove physical PDF
    if file_path.exists():

        file_path.unlink()

    # Remove registry entry
    document_registry.remove(filename)

    # Rebuild vector store from
    # remaining physical documents
    all_documents = []

    for document_info in document_registry.get_all():

        current_filename = document_info["filename"]

        current_path = UPLOAD_DIR / current_filename

        if not current_path.exists():
            continue

        pages = document_service.extract_pages(str(current_path))

        chunks = document_service.create_chunks(pages=pages, filename=current_filename)

        all_documents.extend(chunks)

    rag_service.rebuild(all_documents)

    return {"message": (f"Document '{filename}' " "deleted successfully")}
