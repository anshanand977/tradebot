"""
Database Engine & Session Management
======================================
Single source of truth for DB initialization and session lifecycle.
"""

from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from app.config import settings
from app.db.models import Base


# ─── Engine ───────────────────────────────────────────────────────────────────
engine = create_engine(
    settings.DB_URL,
    connect_args={
        "check_same_thread": False,   # Required for SQLite in multi-threaded use
        "timeout": 30,
    },
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG,
)


# Enable WAL mode for SQLite (much better concurrent read performance)
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")   # 64 MB cache
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't exist and run migrations for missing columns."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at {}", settings.DB_PATH)

    # Dynamic self-healing migration for backwards compatibility
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info(trades)"))
            columns = [row[1] for row in res.fetchall()]
            
            missing_cols = {
                "exchange": "VARCHAR(10) DEFAULT 'NSE'",
                "timeframe": "VARCHAR(5) DEFAULT '1d'",
                "coins_used": "FLOAT",
                "coins_remaining": "FLOAT",
                "ai_reason": "TEXT",
                "notes": "TEXT"
            }
            
            for col_name, col_type in missing_cols.items():
                if col_name not in columns:
                    logger.info("Adding missing column '{}' to trades table...", col_name)
                    conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}"))
            # Note: SQLAlchemy 2.0 connection commit
            try:
                conn.commit()
            except Exception:
                pass
    except Exception as e:
        logger.warning("Database self-healing migration warning: {}", e)


def get_db() -> Session:
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


import threading
import time
import random
from sqlalchemy.exc import OperationalError

db_lock = threading.RLock()


@contextmanager
def db_session():
    """Context manager for use outside FastAPI (background workers, etc.)."""
    with db_lock:
        session = SessionLocal()
        try:
            yield session
            
            # Commit with retry logic for SQLite database lock conflicts
            retries = 5
            backoff = 0.1
            while retries > 0:
                try:
                    session.commit()
                    break
                except OperationalError as oe:
                    if "locked" in str(oe).lower() and retries > 1:
                        retries -= 1
                        time.sleep(backoff + random.uniform(0.05, 0.15))
                        backoff *= 2
                    else:
                        raise
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
