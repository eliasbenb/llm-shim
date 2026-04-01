"""Chat completion API endpoint."""

from fastapi import APIRouter

__all__ = ["router"]

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def create_chat_completion(): ...
