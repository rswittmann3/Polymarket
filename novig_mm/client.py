"""Read-only Novig HTTP client.

Deliberately has **no** order, cancel or transfer methods: this package records and simulates, it does not trade.
Public data uses the unsigned `/v3/public/...` routes (no key needed, throttled per IP, CDN-cached ~15 s).
Signed `GET`s and the signed `POST /v3/echo` credential check are available once keys are configured.

Configuration (environment or `.env`; keys never go on the command line or into logs):
    NOVIG_HOST     default https://api.novig.com   (paper: https://api.paper.novig.com)
    NOVIG_KEY_ID   key UUID
    NOVIG_PEM      path to an unencrypted PKCS#8 PEM, or the PEM text itself
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode, quote

import httpx

from . import signing

PROD = "https://api.novig.com"
PAPER = "https://api.paper.novig.com"
USER_AGENT = "novig-mm-recorder/0.1 (read-only)"


class NovigError(RuntimeError):
    def __init__(self, status: int, code: str | None, message: str):
        super().__init__(f"{status} {code or ''} {message}".strip())
        self.status, self.code = status, code


@dataclass
class NovigConfig:
    host: str = PROD
    key_id: str | None = None
    pem: bytes | None = None

    def __repr__(self) -> str:                       # never print key material
        return f"NovigConfig(host={self.host!r}, key_id={'set' if self.key_id else None}, pem={'set' if self.pem else None})"


def load_config(env: dict[str, str] | None = None) -> NovigConfig:
    if env is None:
        from pm_edges.config import load_dotenv
        load_dotenv()
        env = os.environ                              # type: ignore[assignment]
    host = (env.get("NOVIG_HOST") or PROD).rstrip("/")
    pem_raw = env.get("NOVIG_PEM")
    pem: bytes | None = None
    if pem_raw:
        pem = Path(pem_raw).read_bytes() if "BEGIN" not in pem_raw else pem_raw.encode()
    return NovigConfig(host=host, key_id=env.get("NOVIG_KEY_ID") or None, pem=pem)


class Bucket:
    """Client-side pacing so we stay under the server's token buckets (base tier: read 16 Hz)."""

    def __init__(self, rate_hz: float, burst: float | None = None, clock=time.monotonic, sleep=time.sleep):
        self.rate, self.cap = rate_hz, burst if burst is not None else rate_hz
        self.tokens, self.t = self.cap, clock()
        self.clock, self.sleep = clock, sleep

    def take(self, cost: float = 1.0) -> None:
        now = self.clock()
        self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens < cost:
            wait = (cost - self.tokens) / self.rate
            self.sleep(wait)
            self.tokens, self.t = 0.0, self.clock()
        else:
            self.tokens -= cost


class NovigClient:
    def __init__(self, cfg: NovigConfig | None = None, http: httpx.Client | None = None,
                 read_rate_hz: float = 4.0, sleep=time.sleep):
        self.cfg = cfg or NovigConfig()
        self.http = http or httpx.Client(timeout=20.0, headers={"User-Agent": USER_AGENT})
        self._bucket = Bucket(read_rate_hz, sleep=sleep)
        self._sleep = sleep
        self._key = signing.load_private_key(self.cfg.pem) if self.cfg.pem else None

    # -- core -------------------------------------------------------------------------------------------------
    def _request(self, method: str, path: str, params: dict[str, Any] | None = None,
                 body: bytes = b"", signed: bool = False, retries: int = 3) -> Any:
        raw_query = urlencode({k: v for k, v in (params or {}).items() if v is not None}, quote_via=quote)
        url = f"{self.cfg.host}{path}" + (f"?{raw_query}" if raw_query else "")
        for attempt in range(retries + 1):
            headers: dict[str, str] = {}
            if signed:
                if not (self._key and self.cfg.key_id):
                    raise NovigError(0, "NO_KEY", "signed request needs NOVIG_KEY_ID and NOVIG_PEM")
                headers = signing.signed_headers(self.cfg.key_id, self._key, method, path, raw_query, body)
            if body:
                headers["Content-Type"] = "application/json"
            self._bucket.take()
            try:
                r = self.http.request(method, url, headers=headers, content=body or None)
            except httpx.TransportError as e:
                if attempt == retries:
                    raise NovigError(0, "TRANSPORT", str(e)) from e
                self._sleep(min(2 ** attempt, 8))
                continue
            if r.status_code == 429 and attempt < retries:
                self._sleep(min(float(r.headers.get("Retry-After", "1")), 30.0))
                continue
            if r.status_code >= 500 and attempt < retries:
                self._sleep(min(2 ** attempt, 8))
                continue
            if r.status_code >= 400:
                try:
                    j = r.json()
                    raise NovigError(r.status_code, j.get("code"), str(j.get("message", ""))[:200])
                except ValueError:
                    raise NovigError(r.status_code, None, f"non-JSON error body ({r.headers.get('content-type')})")
            return r.json() if r.content else None
        raise NovigError(0, "RETRIES", "exhausted")      # pragma: no cover

    def get_public(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not path.startswith("/v3/public/"):
            raise ValueError("public routes live under /v3/public/")
        return self._request("GET", path, params)

    def get_signed(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params, signed=True)

    def echo(self, payload: dict[str, Any] | None = None) -> Any:
        """Signed `POST /v3/echo`: a 200 that returns the body means host, key, clock and signature are all correct."""
        body = json.dumps(payload or {"ping": "novig-mm"}, separators=(",", ":")).encode()
        return self._request("POST", "/v3/echo", body=body, signed=True)

    # -- catalog (public) ---------------------------------------------------------------------------------------
    def pages(self, path: str, params: dict[str, Any] | None = None, max_pages: int = 50) -> Iterator[dict[str, Any]]:
        p = dict(params or {})
        for _ in range(max_pages):
            page = self.get_public(path, p)
            yield from page.get("items", [])
            if not page.get("next"):
                return
            p["after"] = page["next"]

    def events(self, league: str, status: str = "OPEN_PREGAME", starts_before_ms: int | None = None,
               starts_after_ms: int | None = None) -> list[dict[str, Any]]:
        return list(self.pages("/v3/public/catalog/events", {
            "league": league, "status": status, "startsBefore": starts_before_ms,
            "startsAfter": starts_after_ms, "limit": 5000}))

    def markets(self, league: str, market_types: str | None = None, event_status: str = "OPEN_PREGAME",
                starts_before_ms: int | None = None, starts_after_ms: int | None = None) -> list[dict[str, Any]]:
        return list(self.pages("/v3/public/catalog/markets", {
            "league": league, "marketType": market_types, "eventStatus": event_status,
            "startsBefore": starts_before_ms, "startsAfter": starts_after_ms, "limit": 5000}))

    def book(self, market_id: str, depth: int = 20) -> dict[str, Any]:
        return self.get_public(f"/v3/public/catalog/markets/{market_id}/book", {"depth": depth})
