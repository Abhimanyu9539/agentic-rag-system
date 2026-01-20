-- file_registry
-- Tracks each ingested PDF: its hash, processing status, and Pinecone vector counts.
-- Used by sync.py to detect NEW / CHANGED / UNCHANGED / DELETED files across runs.

CREATE TABLE IF NOT EXISTS file_registry (
    file_name    TEXT PRIMARY KEY,
    file_hash    TEXT NOT NULL,
    status       TEXT NOT NULL,            -- 'in-progress' | 'active' | 'inactive'
    chunk_count  INTEGER,
    vector_count INTEGER,
    indexed_at   TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ DEFAULT now()
);
