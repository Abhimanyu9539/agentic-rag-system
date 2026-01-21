"""Centralized environment configuration.

All environment variables used by the project must be accessed through
`settings` rather than calling `os.getenv` in module logic. Loading
`.env` is done exactly once, here, on import.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return value


class _Settings:
    @property
    def openai_api_key(self) -> str:
        return _require("OPENAI_API_KEY")

    @property
    def pinecone_api_key(self) -> str:
        return _require("PINECONE_API_KEY")

    @property
    def pinecone_cloud(self) -> str:
        return os.getenv("PINECONE_CLOUD", "aws")

    @property
    def pinecone_region(self) -> str:
        return os.getenv("PINECONE_REGION", "us-east-1")

    @property
    def supabase_url(self) -> str:
        return _require("SUPABASE_URL")

    @property
    def supabase_service_key(self) -> str:
        return _require("SUPABASE_SERVICE_KEY")


settings = _Settings()
