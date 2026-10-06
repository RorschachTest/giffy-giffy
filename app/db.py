from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import psycopg
from psycopg.rows import dict_row

from . import config

SCHEMA = Path(__file__).with_name("schema.sql")


def vec(values: Iterable[float]) -> str:
    """Format a list of floats as a pgvector literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def connect(retries: int = 30) -> psycopg.Connection:
    last: Exception | None = None
    for _ in range(retries):
        try:
            return psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
        except psycopg.OperationalError as exc:  # database container still starting
            last = exc
            time.sleep(1)
    raise RuntimeError(f"could not reach the database: {last}")


@contextmanager
def session() -> Iterator[psycopg.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    sql = SCHEMA.read_text().replace("{EMBED_DIM}", str(config.EMBED_DIM)).replace(
        "{FACE_DIM}", str(config.FACE_DIM)
    )
    with session() as conn:
        # api and worker start together; the lock stops them racing on CREATE.
        conn.execute("SELECT pg_advisory_xact_lock(727274)")
        conn.execute(sql)
