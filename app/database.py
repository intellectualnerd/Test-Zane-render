from __future__ import annotations

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# DB_CONFIG = {
#     "host": "ep-morning-lake-aetbhd21-pooler.c-2.us-east-2.aws.neon.tech",
#     "port": 5432,
#     "user": "neondb_owner",
#     "password": "npg_fjJlOyZh95oE",
#     "database": "neondb",
# }

# Build the connection URL
# DATABASE_URL = (
#     f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
#     f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# )

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger("database")

DATABASE_URL = os.getenv("DATABASE_URL")
logger.info("Loading DATABASE_URL and configuring SQLAlchemy engine")

# Normalize driver to psycopg (v3) to avoid psycopg2 build issues on some platforms
if DATABASE_URL and DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    # If no explicit driver, prefer psycopg v3
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    # Import models to register metadata before create_all
    from .utils import models  # noqa: F401
    logger.info("Creating database tables if not exist")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured")


# Dependency for FastAPI routes
from typing import Generator

def get_db() -> Generator:
    db = SessionLocal()
    logger.debug("DB session opened")
    try:
        yield db
    finally:
        db.close()
        logger.debug("DB session closed")

