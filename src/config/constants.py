import re

# Separator injected by OpenDataLoader into markdown output (Java substitutes the page number).
MD_PAGE_SEPARATOR = "\n<<<ODL_PAGE_BREAK_%page-number%>>>\n"

# Pattern to extract the page number from the ODL separator in markdown text.
ODL_PAGE_PATTERN = re.compile(r"<<<ODL_PAGE_BREAK_(\d+)>>>")

# Working sentinel used during chunk splitting; stripped before final output.
SENTINEL_RE = re.compile(r"<!-- page:(\d+) -->")

# ---------------------------------------------------------------------------
# Vector DB Batch Size (Pinecone recommends <=100 vectors per upsert)
_BATCH_SIZE = 100