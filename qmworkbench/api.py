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
from fastapi import Depends, FastAPI, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError

from . import __version__
from .analysis import analyze, prepare_analysis, get_splits
from .data import MAX_BYTES, demo_data, parse_csv, quality, build_target, demo_pca_data
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


def read_dataset(session, dataset_id, lock=False):
    obj = session.scalar(select(Dataset).where(Dataset.id == dataset_id).with_for_update()) if lock else session.get(Dataset, dataset_id)
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


@app.post("/datasets/demo-pca", dependencies=[Depends(authenticate)])
def demo_pca():
    return persist_dataset(demo_pca_data().to_csv(index=False).encode(), "synthetic-pca-sensors.csv")


@app.get("/datasets/{dataset_id}", dependencies=[Depends(authenticate)])
def dataset(dataset_id: str):
    with session_factory()() as session:
        obj = read_dataset(session, dataset_id)
        return {"id": obj.id, "name": obj.name, "digest": obj.digest, "profile": obj.profile}


@app.get("/datasets", dependencies=[Depends(authenticate)])
def list_datasets(limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0)):
    with session_factory()() as session:
        counts = select(AnalysisRun.dataset_id, func.count().label("runs")).group_by(AnalysisRun.dataset_id).subquery()
        rows = session.execute(select(Dataset.id, Dataset.name, Dataset.created_at, Dataset.profile,
            func.coalesce(counts.c.runs, 0)).outerjoin(counts, counts.c.dataset_id == Dataset.id)
            .order_by(Dataset.created_at.desc(), Dataset.id).offset(offset).limit(limit)).all()
        return {"total": session.scalar(select(func.count()).select_from(Dataset)), "items": [
            {"id": id_, "name": name, "created_at": iso_utc(created), "rows": profile["rows"],
             "columns": len(profile["columns"]), "run_count": count} for id_, name, created, profile, count in rows]}


@app.get("/datasets/{dataset_id}/analyses", dependencies=[Depends(authenticate)])
def list_runs(dataset_id: str, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0)):
    with session_factory()() as session:
        read_dataset(session, dataset_id)
        rows = session.execute(select(AnalysisRun.id, AnalysisRun.created_at,
            AnalysisRun.result["config"]["task"].as_string(), AnalysisRun.result["config"]["target"].as_string())
            .where(AnalysisRun.dataset_id == dataset_id).order_by(AnalysisRun.created_at.desc(), AnalysisRun.id)
            .offset(offset).limit(limit)).all()
        total = session.scalar(select(func.count()).select_from(AnalysisRun).where(AnalysisRun.dataset_id == dataset_id))
        return {"total": total, "items": [{"id": id_, "created_at": iso_utc(created), "task": task, "target": target}
            for id_, created, task, target in rows]}


@app.delete("/analyses/{run_id}", dependencies=[Depends(authenticate)])
def delete_run(run_id: str):
    if not analysis_lock.acquire(blocking=False):
        raise HTTPException(409, "Eine Analyse oder Löschung läuft bereits. Bitte später erneut versuchen.")
    try:
        with session_factory()() as session:
            run = fetch_run(session, run_id)
            # The same parent-row lock is used by training and dataset deletion.
            read_dataset(session, run.dataset_id, lock=True)
            removed = session.execute(delete(AnalysisRun).where(AnalysisRun.id == run_id)).rowcount
            if not removed:
                raise HTTPException(404, "Analyselauf nicht gefunden.")
            session.commit()
            return {"deleted_run_id": run_id}
    finally:
        analysis_lock.release()


@app.delete("/datasets/{dataset_id}", dependencies=[Depends(authenticate)])
def delete_dataset(dataset_id: str, expected_runs: int = Query(..., ge=0)):
    if not analysis_lock.acquire(blocking=False):
        raise HTTPException(409, "Eine Analyse oder Löschung läuft bereits. Bitte später erneut versuchen.")
    try:
        with session_factory()() as session:
            obj = read_dataset(session, dataset_id, lock=True)
            count = session.scalar(select(func.count()).select_from(AnalysisRun).where(AnalysisRun.dataset_id == dataset_id))
            if count != expected_runs:
                raise HTTPException(409, "Die Anzahl der Analyseläufe hat sich geändert. Übersicht aktualisieren und erneut bestätigen.")
            session.execute(delete(AnalysisRun).where(AnalysisRun.dataset_id == dataset_id))
            session.delete(obj)
            session.commit()
            return {"deleted_dataset_id": dataset_id, "deleted_runs": count}
    finally:
        analysis_lock.release()


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


@app.post("/datasets/{dataset_id}/preview", dependencies=[Depends(authenticate)])
def preview_analysis(dataset_id: str, config: AnalysisConfig):
    """Target counts and split checks without fitting models or saving a run."""
    with session_factory()() as session:
        obj = read_dataset(session, dataset_id)
        frame = pd.read_json(io.StringIO(obj.data), orient="split", convert_dates=False)
        try:
            X, y, _ = prepare_analysis(frame, config)
            train, test, _ = get_splits(X, y, config)
            target = build_target(frame, config)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise HTTPException(422, str(exc))
        return {"rows": len(frame), "unknown_target": int(target.isna().sum()),
                "eligible": len(y), "excluded": len(frame) - len(y),
                "positive": int((target == 1).sum()) if config.task == "logistic" else None,
                "negative": int((target == 0).sum()) if config.task == "logistic" else None,
                "n_train": len(train), "n_test": len(test)}


@app.post("/datasets/{dataset_id}/analyses", dependencies=[Depends(authenticate)])
def run_analysis(dataset_id: str, config: AnalysisConfig):
    if not analysis_lock.acquire(blocking=False):
        raise HTTPException(429, "Eine Analyse läuft bereits. Bitte nach Abschluss erneut versuchen.")
    try:
        with session_factory()() as session:
            obj = read_dataset(session, dataset_id, lock=True)
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
            return {"id": run.id, "dataset_id": dataset_id, "result": result, "created_at": iso_utc(run.created_at)}
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
        return {"id": run.id, "dataset_id": run.dataset_id, "result": run.result, "created_at": iso_utc(run.created_at)}


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
