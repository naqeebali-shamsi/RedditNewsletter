-- pgvector extension initialization
-- This script runs automatically on first container start via
-- docker-entrypoint-initdb.d. It will NOT re-run on subsequent starts
-- (PostgreSQL skips initdb when data directory already exists).
--
-- Table creation is handled by SQLAlchemy (models.py) via init_db(),
-- so only the extension setup belongs here.

CREATE EXTENSION IF NOT EXISTS vector;
