-- Local development bootstrap for the OmniSense PostgreSQL instance.
-- Runs once, on first container start, before any Alembic migration.
-- Schema itself is owned by Alembic - only extensions and roles belong here.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";      -- fuzzy entity name matching
CREATE EXTENSION IF NOT EXISTS "btree_gin";    -- composite metadata filters
CREATE EXTENSION IF NOT EXISTS "unaccent";     -- accent-insensitive search

-- Schemas: application tables vs. LangGraph checkpoint tables.
CREATE SCHEMA IF NOT EXISTS omnisense AUTHORIZATION omnisense;
CREATE SCHEMA IF NOT EXISTS checkpoints AUTHORIZATION omnisense;
