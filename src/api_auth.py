from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src import config as config_module
from src.namespace import ApiClient, parse_token_scopes

_security = HTTPBearer(auto_error=False)


def _legacy_client(token: str) -> ApiClient:
	"""Full-access client for the legacy shared bearer token."""
	return ApiClient(
		token=token,
		write_namespace=None,
		read_namespaces=None,
	)


def verify_api_key(
	credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> ApiClient:
	"""Validate Bearer token and return the caller's knowledge-space scope."""
	# Read via module attribute so tests that reload src.config stay consistent.
	config = config_module.config
	expected_legacy = config.API_BEARER_TOKEN.strip()
	scopes = parse_token_scopes(getattr(config, "API_TOKEN_SCOPES_RAW", "") or "")

	if not expected_legacy and not scopes:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="API_BEARER_TOKEN is not configured",
		)

	if not credentials or credentials.scheme.lower() != "bearer":
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Unauthorized",
		)

	presented = credentials.credentials
	if presented in scopes:
		return scopes[presented]
	if expected_legacy and presented == expected_legacy:
		return _legacy_client(presented)

	raise HTTPException(
		status_code=status.HTTP_401_UNAUTHORIZED,
		detail="Unauthorized",
	)
