"""Embeddings API endpoint."""

from fastapi import APIRouter

__all__ = ["router"]

router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings")
async def create_embeddings(): ...
