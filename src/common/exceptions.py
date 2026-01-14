class AppError(Exception):
    """Base exception for project-level application errors."""


class AdapterInitializationError(AppError):
    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self.component = component
        self.provider = provider
        self.model = model
        super().__init__(message)
