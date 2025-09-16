from typing import Dict

from fastapi import APIRouter

router = APIRouter(prefix="/indices", tags=["indices"])


@router.get("/")
async def list_indices() -> Dict[str, str]:
    # Placeholder: return some sample indices names
    return {"indices": "sp500, nasdaq, dow"}


@router.get("/health")
async def indices_health() -> Dict[str, str]:
    return {"status": "ok"}
