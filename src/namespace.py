"""Knowledge-space (Ruang Pengetahuan) helpers.

Points carry ``metadata.namespace``. Token scopes decide which spaces a
client may write and read. Legacy ``API_BEARER_TOKEN`` keeps full access
(no read filter) so existing Telegram/frontend clients keep working.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)

NAMESPACE_PAYLOAD_KEY = "metadata.namespace"


@dataclass(frozen=True, slots=True)
class ApiClient:
    """Authenticated API caller with optional knowledge-space scope."""

    token: str
    write_namespace: str | None
    """Namespace stamped on ingest. None = use DEFAULT_WRITE_NAMESPACE."""

    read_namespaces: tuple[str, ...] | None
    """Namespaces visible on retrieve. None = full access (no filter)."""

    @property
    def is_scoped(self) -> bool:
        return self.read_namespaces is not None


def parse_token_scopes(raw: str) -> dict[str, ApiClient]:
    """Parse ``API_TOKEN_SCOPES`` JSON into token → ApiClient.

    Expected shape::

        {
          "ppid-token": {
            "write": "ppid",
            "read": ["ppid", "tabalong-umum"]
          }
        }
    """
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("API_TOKEN_SCOPES is not valid JSON: %s", exc)
        return {}
    if not isinstance(data, dict):
        logger.error("API_TOKEN_SCOPES must be a JSON object")
        return {}

    out: dict[str, ApiClient] = {}
    for token, spec in data.items():
        if not isinstance(token, str) or not token.strip():
            continue
        if not isinstance(spec, dict):
            logger.warning("Ignoring malformed scope for token %r", token[:8])
            continue
        write = spec.get("write")
        read = spec.get("read")
        if not isinstance(write, str) or not write.strip():
            logger.warning("Ignoring scope for token %r: missing write", token[:8])
            continue
        if not isinstance(read, list) or not read:
            logger.warning("Ignoring scope for token %r: missing read list", token[:8])
            continue
        read_ns = tuple(str(x).strip() for x in read if str(x).strip())
        if not read_ns:
            continue
        out[token.strip()] = ApiClient(
            token=token.strip(),
            write_namespace=write.strip(),
            read_namespaces=read_ns,
        )
    return out


def resolve_write_namespace(client: ApiClient, default: str) -> str:
    """Namespace to stamp on new chunks for this client."""
    if client.write_namespace:
        return client.write_namespace
    return (default or "").strip()


def build_namespace_filter(
    read_namespaces: tuple[str, ...] | list[str] | None,
) -> rest.Filter | None:
    """Qdrant filter matching any of the given namespaces.

    Returns None when ``read_namespaces`` is None (full access / legacy).
    Empty list is treated as a filter that matches nothing.
    """
    if read_namespaces is None:
        return None
    values = [str(v).strip() for v in read_namespaces if str(v).strip()]
    if not values:
        return rest.Filter(
            must=[
                rest.FieldCondition(
                    key=NAMESPACE_PAYLOAD_KEY,
                    match=rest.MatchValue(value="__no_such_namespace__"),
                )
            ]
        )
    if len(values) == 1:
        return rest.Filter(
            must=[
                rest.FieldCondition(
                    key=NAMESPACE_PAYLOAD_KEY,
                    match=rest.MatchValue(value=values[0]),
                )
            ]
        )
    return rest.Filter(
        must=[
            rest.FieldCondition(
                key=NAMESPACE_PAYLOAD_KEY,
                match=rest.MatchAny(any=values),
            )
        ]
    )


def merge_filters(
    *filters: rest.Filter | None,
) -> rest.Filter | None:
    """Combine Qdrant filters with AND semantics. Drops Nones."""
    parts = [f for f in filters if f is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    must: list = []
    for f in parts:
        if f.must:
            must.extend(f.must)
        if f.should or f.must_not or f.min_should:
            # Nest complex filters so we don't flatten should/must_not wrongly.
            must.append(rest.Filter(must=f.must, should=f.should, must_not=f.must_not))
    return rest.Filter(must=must)
