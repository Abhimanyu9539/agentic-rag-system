from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from src.common.exceptions import AdapterInitializationError
from src.common.logging import get_logger

load_dotenv()
logger = get_logger(__name__)


def get_chat_model(
    model: str | None = None,
    model_provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    logger.info("Initializing chat model provider=%s model=%s", model_provider, model)

    try:
        chat_model = init_chat_model(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
        )
    except Exception as exc:
        logger.exception(
            "Failed to initialize chat model provider=%s model=%s",
            model_provider,
            model,
        )
        raise AdapterInitializationError(
            "Unable to initialize chat model.",
            component="llm",
            provider=model_provider,
            model=model,
        ) from exc

    logger.info("Chat model initialized provider=%s model=%s", model_provider, model)
    return chat_model


if __name__ == "__main__":
    chat_model = get_chat_model(model="gpt-5.1", model_provider="openai")
    print(chat_model)
    