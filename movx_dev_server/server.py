"""movx dev server: the V1 pipeline behind a small HTTP API, plus the Flutter web build at /.

    uvicorn server:app --host 0.0.0.0 --port 7860

Env: MOVX_MODELS (default ./models), MOVX_DATA (default /data if it exists, else ./data),
MOVX_WEB (default ./web), MOVX_THREADS (default: all cores).
"""
import json
import os
import queue
import shutil
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import movx_pipeline as P

HERE = Path(__file__).parent
MODELS = Path(os.environ.get("MOVX_MODELS", HERE / "models"))
DATA = Path(os.environ.get("MOVX_DATA", "/data" if Path("/data").is_dir() else HERE / "data"))
WEB = Path(os.environ.get("MOVX_WEB", HERE / "web"))
ANALYSES, UPLOADS = DATA / "analyses", DATA / "uploads"
ANALYSES.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
PASS2_MODELS = {"yolo26x-pose_int8w.tflite", "yolo26n-pose.tflite"}

analyzer = P.Analyzer(MODELS, threads=int(os.environ.get("MOVX_THREADS", os.cpu_count() or 4)))
runs = {}                 # id -> {state, stage, fraction, message, fix, queued}
jobs = queue.Queue()
lock = threading.Lock()


def worker():
    while True:
        rid, video, pass2, source = jobs.get()
        run = runs[rid]
        out = ANALYSES / rid

        def progress(stage, f):
            run.update(stage=stage, fraction=round(f, 3))
        try:
            if run.get("cancel"):
                raise P.Cancelled()
            run.update(state="running", stage="Starting", fraction=0)
            analyzer.analyze(video, out, pass2, source=source, progress=progress,
                             cancelled=lambda: run.get("cancel", False))
            run.update(state="done")
        except P.Rejected as e:
            shutil.rmtree(out, ignore_errors=True)
            run.update(state="rejected", message=e.reason, fix=e.fix)
        except P.Cancelled:
            shutil.rmtree(out, ignore_errors=True)
            run.update(state="cancelled")
        except Exception as e:
            traceback.print_exc()
            shutil.rmtree(out, ignore_errors=True)
            run.update(state="failed", message=f"{type(e).__name__}: {e}")
        finally:
            Path(video).unlink(missing_ok=True)


threading.Thread(target=worker, daemon=True).start()
app = FastAPI(title="movx dev server")


def _dir(aid):
    d = ANALYSES / Path(aid).name
    if not (d / "result.json").is_file():
        raise HTTPException(404, "no such analysis")
    return d


@app.post("/api/analyze")
def analyze(video: UploadFile = File(...), pass2Model: str = Form("yolo26x-pose_int8w.tflite")):
    if pass2Model not in PASS2_MODELS:
        raise HTTPException(400, "unknown model")
    with lock:
        rid = time.strftime("%Y%m%d_%H%M%S")
        while rid in runs or (ANALYSES / rid).exists():
            rid += "_"
        runs[rid] = {"state": "queued", "stage": "Waiting for the server", "fraction": 0}
    path = UPLOADS / f"{rid}{Path(video.filename or 'v.mp4').suffix or '.mp4'}"
    with open(path, "wb") as f:
        shutil.copyfileobj(video.file, f, 1 << 20)
    jobs.put((rid, path, pass2Model, video.filename or "video"))
    return {"id": rid}


@app.get("/api/runs/{rid}")
def run_state(rid: str):
    r = runs.get(rid)
    if r is None:
        raise HTTPException(404, "no such run")
    return {k: v for k, v in r.items() if k != "cancel"}


@app.post("/api/runs/{rid}/cancel")
def cancel(rid: str):
    if rid in runs:
        runs[rid]["cancel"] = True
    return {}


@app.get("/api/analyses")
def list_analyses():
    out = []
    for d in ANALYSES.iterdir():
        try:
            out.append(json.loads((d / "result.json").read_text()))
        except Exception:
            pass
    return sorted(out, key=lambda a: -a["createdAt"])


@app.get("/api/analyses/{aid}")
def get_analysis(aid: str):
    return json.loads((_dir(aid) / "result.json").read_text())


@app.put("/api/analyses/{aid}")
def save_analysis(aid: str, body: dict):
    d = _dir(aid)
    old = json.loads((d / "result.json").read_text())
    for k in ("correctedAh", "correctedBr", "bowler"):         # the only fields the app edits
        if k in body:
            old[k] = body[k]
    (d / "result.json").write_text(json.dumps(old))
    return old


@app.delete("/api/analyses/{aid}")
def delete_analysis(aid: str):
    shutil.rmtree(_dir(aid))
    return {}


@app.get("/api/analyses/{aid}/frames/{name}")
def frame(aid: str, name: str):
    f = _dir(aid) / "frames" / Path(name).name
    if not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


if WEB.is_dir():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
