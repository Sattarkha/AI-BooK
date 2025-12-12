from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import uuid

router = APIRouter()

# Request/Response models
class EmbeddingRequest(BaseModel):
    text: str
    model: str = "text-embedding-ada-002"


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    text: str
    model: str


@router.post("/embed", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """
    Create embeddings for the given text
    """
    # In a real implementation, this would call the OpenAI embeddings API
    # For now, return a mock embedding (1536-dimensional vector for text-embedding-ada-002)
    mock_embedding = [0.0] * 1536  # Placeholder for actual embedding

    return EmbeddingResponse(
        embedding=mock_embedding,
        text=request.text,
        model=request.model
    )


class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    model: str = "text-embedding-ada-002"


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str


@router.post("/embed-batch", response_model=BatchEmbeddingResponse)
async def create_batch_embedding(request: BatchEmbeddingRequest):
    """
    Create embeddings for multiple texts
    """
    # In a real implementation, this would call the OpenAI embeddings API
    # For now, return mock embeddings
    mock_embeddings = [[0.0] * 1536 for _ in request.texts]

    return BatchEmbeddingResponse(
        embeddings=mock_embeddings,
        model=request.model
    )