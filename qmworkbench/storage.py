import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy import JSON, DateTime, ForeignKey, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def identity():
    return str(uuid.uuid4())


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identity)
    name: Mapped[str] = mapped_column(String(200))
    digest: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    data: Mapped[str] = mapped_column(Text)
    profile: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=identity)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    result: Mapped[dict] = mapped_column(JSON)
    artifact: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


@lru_cache
def engine():
    url = os.getenv("DATABASE_URL", "sqlite:///./workbench.db")
    if url.startswith(("postgres://", "postgresql://")):
        url = "postgresql+psycopg://" + url.split("://", 1)[1]
    return create_engine(url, pool_pre_ping=True, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})


def session_factory():
    return sessionmaker(engine(), expire_on_commit=False)
