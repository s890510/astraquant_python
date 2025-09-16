from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(prefix="/features", tags=["features"])


class BuildRequest(BaseModel):
    dataset: Optional[str] = "default"


class BuildResponse(BaseModel):
    status: str
    output_path: str


@router.post("/build", response_model=BuildResponse)
async def build_features(req: BuildRequest) -> BuildResponse:
    # Dummy feature building: write an empty marker file into data dir
    out_dir = Path(settings.DATA_DIR) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"features_{req.dataset}.parquet"
    # Create an empty file as placeholder
    if not out_file.exists():
        out_file.touch()
    return BuildResponse(status="ok", output_path=str(out_file))
