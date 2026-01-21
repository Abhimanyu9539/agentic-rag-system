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
SUPABASE_FILE_REGISTRY_TABLE  = "file_registry"
SUPABASE_IMAGE_REGISTRY_TABLE = "image_registry"
SUPABASE_IMAGES_BUCKET        = "table-images"

# ---------------------------------------------------------------------------
# Table extraction (PyMuPDF / OpenDataLoader)
# ---------------------------------------------------------------------------
# Number of text blocks above a table to include as same-page heading context.
TABLE_CONTEXT_NUM_BLOCKS = 4
# Max vertical distance (pt) between table top and a candidate context block.
TABLE_CONTEXT_MAX_DISTANCE_PT = 250
# Tolerance (pt) when checking that a block sits above the table top.
TABLE_CONTEXT_Y_TOLERANCE_PT = 2
# Blocks whose top is within this distance of the page top are treated as
# header/footer noise and excluded from context.
TABLE_TEXT_BLOCK_MIN_Y_PT = 15
# A table whose top edge falls within this distance of the page top is treated
# as a continuation candidate (no room above for a same-page heading).
TABLE_PAGE_TOP_THRESHOLD_PT = 100
# Tables smaller than this in either dimension (pt) are considered degenerate.
TABLE_MIN_DIMENSION_PT = 10
# DPI used when rasterizing tables to PNG.
TABLE_RENDER_DPI = 200

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
# Keep parsed JSON/markdown files on disk after processing. Off by default;
# files are cleaned up immediately after reading.
DEBUG_SAVE_PARSED = False

# Also keep extracted table PNGs on disk in addition to uploading them to
# Supabase Storage. Off by default.
DEBUG_SAVE_IMAGES = False