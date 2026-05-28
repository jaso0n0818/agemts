# SPDX-FileCopyrightText: 2025 Rayleigh Research
# SPDX-License-Identifier: MIT
"""Real-time local lab dashboard (FastAPI + SSE)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

SN79_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SN79_ROOT / "agents"))

from lab.metrics import load_state, state_path  # noqa: E402

STATIC = Path(__file__).parent / "static"
REFRESH_INTERVAL_SEC = 10.0
app = FastAPI(title="τaos Local Lab", version="1.0")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text())


@app.get("/api/state")
async def api_state() -> dict:
    return load_state()


@app.get("/api/stream")
async def api_stream():
    async def gen():
        while True:
            try:
                raw = state_path().read_text()
            except OSError:
                raw = "{}"
            yield f"data: {raw}\n\n"
            await asyncio.sleep(REFRESH_INTERVAL_SEC)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "state_path": str(state_path())}


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
