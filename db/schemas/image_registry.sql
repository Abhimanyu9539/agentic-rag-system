-- image_registry
-- Tracks every table image extracted from a PDF and uploaded to Supabase Storage.
-- One row per image. Cascades on file_registry deletion so DB rows clear automatically;
-- the corresponding storage objects must be deleted explicitly via the Storage API.

CREATE TABLE IF NOT EXISTS image_registry (
    id           BIGSERIAL PRIMARY KEY,
    file_name    TEXT NOT NULL REFERENCES file_registry(file_name) ON DELETE CASCADE,
    image_name   TEXT NOT NULL,            -- e.g. 'table_p3_0.png'
    storage_path TEXT NOT NULL UNIQUE,     -- e.g. '{pdf_stem}/table_p3_0.png'
    public_url   TEXT NOT NULL,
    page         INTEGER NOT NULL,
    table_id     INTEGER NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now(),
    UNIQUE (file_name, image_name)
);

CREATE INDEX IF NOT EXISTS idx_image_registry_file_name
    ON image_registry(file_name);
