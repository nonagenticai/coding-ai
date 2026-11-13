"""Outbound client to coding-analysis (the as-built code-graph service).

Used by the P4.1 ``GET /api/ai/architecture/proposals?container=<name>``
endpoint to fetch the per-container Component list, which is then handed
to ``ArchitectureClassifier`` to produce ranked ArchiMate-type
proposals.

Contract pinned in
``requirements_management/docs/ARCHITECTURE_PROPOSALS_CONTRACT.md``:
the primary binding key downstream consumers key on is
``(container, component_id)``, where ``component_id`` is the
``Component.id`` returned here. Don't substitute ``path`` — the
conformance tool's ``_artifact_binding_key`` reads ``component_id``.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from clients.outbound_identity import outbound_headers
from config.settings import get_settings

logger = logging.getLogger(__name__)

# Single shared client per process — mirrors agents/base.py in WM. Avoids
# per-request connection-pool churn for the typical "user pulls a
# container's proposals" interaction (potentially several requests in a
# short polling window).
_client: httpx.AsyncClient | None = None


def _get_client(timeout: float = 5.0) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=timeout)
    return _client


async def close() -> None:
    """Close the shared client. Hook from app shutdown if added later."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


class CodingAnalysisUnavailableError(Exception):
    """Raised when coding-analysis is not configured or unreachable.

    The router catches this and degrades the proposals response to an
    empty ``proposals[]`` (200) rather than 5xx — matches the contract's
    "errors degrade per unit" stance, applied here to the bulk pull
    when the whole upstream is dark.
    """


async def fetch_components(container: str, *, tenant_id: str) -> list[dict[str, Any]]:
    """Fetch the architectural component list for one container.

    Returns the raw ``Component`` dicts (``{id, name, files, dependencies,
    layer, tags}``) from coding-analysis. Caller maps them into the
    proposal contract ``code_unit`` shape.

    - ``[]`` is a valid return (container has no components — not an error).
    - Missing setting OR 404 from upstream → ``CodingAnalysisUnavailableError``
      (the router translates to an empty-but-200 response with a note).
    - Other 4xx/5xx / network errors → ``CodingAnalysisUnavailableError`` too;
      logged with the status code.
    """
    url = (get_settings().coding_analysis_url or "").rstrip("/")
    if not url:
        raise CodingAnalysisUnavailableError("coding_analysis_url is not configured")

    target = f"{url}/api/analysis/architecture/components/{container}"
    client = _get_client()
    try:
        # The tenant is the CUSTOMER's; coding-ai's own license rides alongside
        # it in Authorization so coding-analysis can accept the assertion once it
        # stops trusting a bare X-Tenant-Id. See clients.outbound_identity.
        resp = await client.get(target, headers=outbound_headers(tenant_id))
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
        logger.warning("coding-analysis unreachable for container=%s: %s", container, exc)
        raise CodingAnalysisUnavailableError(f"network error: {exc}") from exc

    if resp.status_code == 404:
        # No graph for this container (yet). The contract says coding-ai
        # never lies about "no proposals" — pass that through as an empty
        # list, not an exception. Equivalent to "the container is known
        # to coding-analysis but has no components".
        return []
    if resp.status_code >= 400:
        logger.warning(
            "coding-analysis returned %d for container=%s: %s",
            resp.status_code,
            container,
            resp.text[:200],
        )
        raise CodingAnalysisUnavailableError(f"coding-analysis returned {resp.status_code}")

    payload = resp.json()
    if not isinstance(payload, list):
        logger.warning(
            "coding-analysis returned non-list payload for container=%s: %r",
            container,
            type(payload).__name__,
        )
        return []
    return [c for c in payload if isinstance(c, dict)]
