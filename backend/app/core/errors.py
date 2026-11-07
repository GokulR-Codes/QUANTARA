"""Centralised exception hierarchy."""
from __future__ import annotations


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class VideoValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class VideoProcessingError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class ModelNotReadyError(AppError):
    def __init__(self, message: str = "No trained model available") -> None:
        super().__init__(message, status_code=503)


class CheckpointError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class DatasetError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)


class AuthenticationError(AppError):
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status_code=401)


class AuthorizationError(AppError):
    def __init__(self, message: str = "Insufficient permissions") -> None:
        super().__init__(message, status_code=403)


class ArtifactError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class JobConflictError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=409)


class JobCancelled(Exception):
    """Raised inside a processing worker when the job has been cancelled by the operator.

    Deliberately *not* an AppError: it is control flow, never an HTTP response.
    """
