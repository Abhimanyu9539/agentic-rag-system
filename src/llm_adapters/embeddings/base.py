from typing import Any

from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from src.common.exceptions import AdapterInitializationError
from src.common.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


def get_embeddings_model(
    model: str,
    model_provider: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    try:
        logger.info(f"Initializing embeddings model provider={model_provider} model={model}")
        embeddings_model = init_embeddings(model=model, provider=model_provider, **kwargs)
    except Exception as exc:
        logger.exception(f"Failed to initialize embeddings model provider={model_provider} model={model}")
        raise AdapterInitializationError(
            "Unable to initialize embeddings model.",
            component="embeddings",
            provider=model_provider,
            model=model,
        ) from exc

    logger.info(f"Embeddings model initialized provider={model_provider} model={model}")
    return embeddings_model


if __name__ == "__main__":
    embeddings_model = get_embeddings_model(model="text-embedding-3-small", model_provider="openai")
    print(embeddings_model)
    print(embeddings_model.embed_query("Hello, world!"))
