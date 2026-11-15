"""Bearer-token check shared by the REST API and the MCP HTTP transport."""

from __future__ import annotations

import hmac
import os


def check_bearer(header: str | None) -> tuple[int, str]:
    """Return (http_status, message). 200 means allowed.

    * DRIFTREV_API_TOKEN set  -> header must be 'Bearer <token>' (constant-time compare).
    * unset                   -> only allowed if DRIFTREV_ALLOW_NO_AUTH=true, else 503
      (fail closed: a forgotten token must not silently expose the service).
    """
    token = os.environ.get("DRIFTREV_API_TOKEN", "")
    if not token:
        if os.environ.get("DRIFTREV_ALLOW_NO_AUTH", "").lower() == "true":
            return 200, "no-auth mode"
        return 503, ("Server auth not configured: set DRIFTREV_API_TOKEN "
                     "(or DRIFTREV_ALLOW_NO_AUTH=true for local development only).")
    scheme, _, supplied = (header or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied:
        return 401, "Missing bearer token."
    # compare bytes: hmac.compare_digest on str raises for non-ASCII input
    if not hmac.compare_digest(supplied.strip().encode(), token.encode()):
        return 401, "Invalid token."
    return 200, "ok"
