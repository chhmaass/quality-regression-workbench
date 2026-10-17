import hmac
import io
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import timezone

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from . import __version__
from .analysis import analyze
from .data import MAX_BYTES, demo_data, parse_csv, quality
from .report import html_report, safe_csv
from .schemas import AnalysisConfig
from .storage import AnalysisRun, Base, Dataset, engine, session_factory

analysis_lock = threading.Lock()


def iso_utc(value):
    return value.replace(tzinfo=timezone.utc).isoformat() if value.tzinfo is None else value.isoformat()


def authenticate(x_api_key: str = Header(default="")):
    expected = os.getenv("WORKBENCH_API_KEY", "")
    if not expected:
        raise HTTPException(503, "WORKBENCH_API_KEY fehlt; kein ungeschützter Datenzugriff.")
    if not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Ungültiger Zugangsschlüssel.")


@asynccontextmanager
async def lifespan(app):
    if os.getenv("AUTO_CREATE_TABLES", "0") == "1":
        Base.metadata.create_all(engine())
    yield


app = FastAPI(title="QM Regression Workbench", version=__version__, lifespan=lifespan)


@app.get("/health")
def health():
    try:
        with engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(503, "Datenbank nicht erreichbar.")
    return {"status": "ok", "version": __version__}


def read_dataset(session, dataset_id):
    obj = session.get(Dataset, dataset_id)
    if not obj:
        raise HTTPException(404, "Datensatz nicht gefunden.")
    return obj


def persist_dataset(raw, name):
    frame, digest = parse_csv(raw)
    with session_factory()() as session:
        existing = session.scalar(select(Dataset).where(Dataset.digest == digest))
        if existing:
            return {"id": existing.id, "name": existing.name, "digest": digest, "reused": True, "profile": existing.profile}
        obj = Dataset(name=name[:200], digest=digest, data=frame.to_json(orient="split", date_format="iso"), profile=quality(frame))
        session.add(obj)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(select(Dataset).where(Dataset.digest == digest))
            if not existing:
                raise
            return {"id": existing.id, "name": existing.name, "digest": digest, "reused": True, "profile": existing.profile}
        return {"id": obj.id, "name": obj.name, "digest": digest, "reused": False, "profile": obj.profile}


@app.post("/datasets", dependencies=[Depends(authenticate)])
async def upload(file: UploadFile):
    try:
        raw = await file.read(MAX_BYTES + 1)
        return persist_dataset(raw, file.filename or "upload.csv")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    finally:
        await file.close()


@app.post("/datasets/demo", dependencies=[Depends(authenticate)])
def demo():
    return persist_dataset(demo_data().to_csv(index=False).encode(), "synthetic-qm.csv")


@app.get("/datasets/{dataset_id}", dependencies=[Depends(authenticate)])
def dataset(dataset_id: str):
    with session_factory()() as session:
        obj = read_dataset(session, dataset_id)
        return {"id": obj.id, "name": obj.name, "digest": obj.digest, "profile": obj.profile}


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@app.post("/datasets/{dataset_id}/analyses", dependencies=[Depends(authenticate)])
def run_analysis(dataset_id: str, config: AnalysisConfig):
    if not analysis_lock.acquire(blocking=False):
        raise HTTPException(429, "Eine Analyse läuft bereits. Bitte nach Abschluss erneut versuchen.")
    try:
        with session_factory()() as session:
            obj = read_dataset(session, dataset_id)
            frame = pd.read_json(io.StringIO(obj.data), orient="split", convert_dates=False)
            try:
                result, artifact = analyze(frame, config)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise HTTPException(422, str(exc))
            result = clean_json(result)
            result.update(dataset_name=obj.name, dataset_digest=obj.digest, dataset_created_at=iso_utc(obj.created_at))
            run = AnalysisRun(dataset_id=dataset_id, result=result, artifact=artifact)
            session.add(run)
            session.commit()
            return {"id": run.id, "result": result, "created_at": iso_utc(run.created_at)}
    except HTTPException:
        raise
    except Exception:
        logging.getLogger(__name__).exception("Analyse fehlgeschlagen")
        raise HTTPException(500, "Analyse fehlgeschlagen; Serverprotokoll prüfen.")
    finally:
        analysis_lock.release()


def fetch_run(session, run_id):
    run = session.get(AnalysisRun, run_id)
    if not run:
        raise HTTPException(404, "Analyselauf nicht gefunden.")
    return run


@app.get("/analyses/{run_id}", dependencies=[Depends(authenticate)])
def get_run(run_id: str):
    with session_factory()() as session:
        run = fetch_run(session, run_id)
        return {"id": run.id, "result": run.result, "created_at": iso_utc(run.created_at)}


@app.get("/analyses/{run_id}/report", dependencies=[Depends(authenticate)], response_class=HTMLResponse)
def report(run_id: str):
    with session_factory()() as session:
        run = fetch_run(session, run_id)
        dataset = read_dataset(session, run.dataset_id)
        return html_report(dict(run.result, created_at=iso_utc(run.created_at)), dataset.name, run.id)


@app.get("/analyses/{run_id}/csv", dependencies=[Depends(authenticate)])
def csv_export(run_id: str, model: str = "reference"):
    with session_factory()() as session:
        run = fetch_run(session, run_id)
        if model not in run.result["models"]:
            raise HTTPException(404, "Modell nicht vorhanden.")
        return Response(safe_csv(run.result["models"][model]["rows"]), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="qm-results.csv"'})


@app.get("/analyses/{run_id}/config", dependencies=[Depends(authenticate)])
def config_export(run_id: str):
    with session_factory()() as session:
        run = fetch_run(session, run_id)
        return Response(json.dumps(run.result["config"], ensure_ascii=False, indent=2), media_type="application/json")
