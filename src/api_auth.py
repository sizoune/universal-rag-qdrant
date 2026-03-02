from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import config

_security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> None:
    """Validate Bearer token for protected API routes."""
    expected = config.API_BEARER_TOKEN.strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_BEARER_TOKEN is not configured",
        )

    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
