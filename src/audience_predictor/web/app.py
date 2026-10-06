"""Coverage desk: upload a script, read the guess."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from audience_predictor.coverage import describe_backend
from audience_predictor.web.demos import demo_catalog, load_demo
from audience_predictor.web.studio import predict_pages

STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 2_000_000


class PredictBody(BaseModel):
    text: str | None = None
    demo_id: str | None = None
    k: int = Field(default=8, ge=3, le=16)


def create_app() -> FastAPI:
    app = FastAPI(title="Audience Predictor", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        judge = describe_backend()
        return {"status": "ok", **judge}

    @app.get("/api/demos")
    def demos() -> list[dict[str, str]]:
        return demo_catalog()

    @app.post("/api/predict")
    def predict(body: PredictBody) -> dict:
        text = (body.text or "").strip()
        meta = None
        if body.demo_id:
            try:
                meta, text = load_demo(body.demo_id)
            except KeyError as exc:
                raise HTTPException(404, f"unknown demo {body.demo_id}") from exc
        if not text:
            raise HTTPException(400, "paste a script, upload a file, or pick a spec")
        try:
            result = predict_pages(text, k=body.k)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"predict failed: {exc}") from exc
        result["demo"] = meta
        result["excerpt"] = _excerpt(text)
        return result

    @app.post("/api/predict-file")
    async def predict_file(file: UploadFile = File(...), k: int = 8) -> dict:
        raw = await file.read()
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "script larger than 2 MB")
        text = raw.decode("utf-8", errors="replace")
        try:
            result = predict_pages(text, k=k)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"predict failed: {exc}") from exc
        result["demo"] = {
            "id": "upload",
            "title": (file.filename or "UNTITLED").upper(),
            "slugline": "INT. UPLOAD - CONTINUOUS",
        }
        result["excerpt"] = _excerpt(text)
        return result

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _excerpt(text: str, n: int = 1400) -> str:
    body = text.strip()
    if len(body) <= n:
        return body
    return body[:n].rstrip() + "\n\n(CONTINUED)"


app = create_app()
