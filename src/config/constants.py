import re

# ---------------------------------------------------------------------------
# Markdown / page tracking
# ---------------------------------------------------------------------------
# Separator injected by OpenDataLoader into markdown output (Java substitutes the page number).
MD_PAGE_SEPARATOR = "\n<<<ODL_PAGE_BREAK_%page-number%>>>\n"

# Pattern to extract the page number from the ODL separator in markdown text.
ODL_PAGE_PATTERN = re.compile(r"<<<ODL_PAGE_BREAK_(\d+)>>>")

# Working sentinel used during chunk splitting; stripped before final output.
SENTINEL_RE = re.compile(r"<!-- page:(\d+) -->")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER = "openai"
EMBEDDING_DIMENSION = 1536

# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
# Pinecone recommends <=100 vectors per upsert.
BATCH_SIZE = 100
PINECONE_METRIC = "cosine"
DEFAULT_PINECONE_INDEX = "agentic-rag-system"

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_FILE_REGISTRY_TABLE = "file_registry"