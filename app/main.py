from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.routers import ml_router, features_router, news_router, indices_router, twse_instflows_router

app = FastAPI(title="AstraQuant API")


@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})


# Include routers
app.include_router(ml_router)
app.include_router(features_router)
app.include_router(news_router)
app.include_router(indices_router)
app.include_router(twse_instflows_router, prefix="/twse")
