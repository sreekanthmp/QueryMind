from fastapi import FastAPI
from src.loaders.confluence.service import \
    DocumentLoaderService

app = FastAPI()
document_service = DocumentLoaderService()


@app.get("/confluence-space")
async def create_vectorstore():
    # Initialize the DocumentLoaderService class
    response = document_service.process_space_documents()
    return response
