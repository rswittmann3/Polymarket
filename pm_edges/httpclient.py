"""JSON HTTP client with retries, raw-response archiving and offline replay.

Every response body is written to ``data/raw/<game>/<run>/<key>.json`` before
it is parsed. With ``offline=True`` the same request is answered from the most
recent saved copy instead of the network, so the whole pipeline can be re-run
from disk.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

# Never written to disk.
SECRET_PARAMS = {"apiKey", "apikey", "api_key"}
# Response headers kept with each saved body.
KEPT_HEADERS = ("x-requests-remaining", "x-requests-used", "x-requests-last",
                "content-type", "date")


class ApiError(RuntimeError):
    pass


class OfflineMiss(ApiError):
    pass


class ClientError(ApiError):
    """A 4xx answer. Saved like a body, so offline runs fail the same way."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def request_key(url: str, params: dict[str, Any] | None) -> str:
    """Stable file name for a request, without any secret parameter."""
    parts = urlsplit(url)
    host = parts.netloc.split(".")[0]
    path = re.sub(r"[^A-Za-z0-9]+", "_", parts.path).strip("_") or "root"
    clean = sorted((k, str(v)) for k, v in (params or {}).items() if k not in SECRET_PARAMS)
    digest = hashlib.sha1(json.dumps(clean).encode()).hexdigest()[:10]
    return f"{host}__{path[:80]}__{digest}"


@dataclass
class Response:
    body: Any
    headers: dict[str, str] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=utcnow)
    from_cache: bool = False


class RawStore:
    """Saves raw responses per game and run, and finds them again offline."""

    def __init__(self, root: Path, game_id: str, run_id: str | None = None):
        self.game_dir = root / game_id
        self.run_id = run_id or utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.game_dir / self.run_id
        self._lock = threading.Lock()

    def save(self, key: str, url: str, params: dict | None, resp: Response | None,
             error: str | None = None) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / f"{key}.json"
        record = {
            "url": url,
            "params": {k: v for k, v in (params or {}).items() if k not in SECRET_PARAMS},
            "fetched_at": (resp.fetched_at if resp else utcnow()).isoformat(),
            "headers": resp.headers if resp else {},
            "body": resp.body if resp else None,
        }
        if error is not None:
            record["error"] = error
        with self._lock:
            path.write_text(json.dumps(record, indent=1))
        return path

    def load(self, key: str, run_id: str | None = None) -> Response:
        runs = [self.game_dir / run_id] if run_id else sorted(
            (p for p in self.game_dir.glob("*") if p.is_dir()), reverse=True)
        for run in runs:
            path = run / f"{key}.json"
            if path.is_file():
                record = json.loads(path.read_text())
                if "error" in record:
                    raise ClientError(record["error"])
                return Response(
                    body=record["body"],
                    headers=record.get("headers", {}),
                    fetched_at=datetime.fromisoformat(record["fetched_at"]),
                    from_cache=True,
                )
        raise OfflineMiss(f"no saved response for {key} under {self.game_dir}")


class JsonClient:
    def __init__(self, store: RawStore, *, offline: bool = False,
                 offline_run: str | None = None, timeout: float = 20.0,
                 retries: int = 3, transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep):
        self.store = store
        self.offline = offline
        self.offline_run = offline_run
        self.retries = retries
        self._sleep = sleep
        self.latest_fetch: datetime | None = None   # newest response time seen
        self._http = None if offline else httpx.Client(
            timeout=timeout, transport=transport, follow_redirects=True,
            headers={"User-Agent": "pm_edges/0.1 (read-only)"})

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def get(self, url: str, params: dict[str, Any] | None = None) -> Response:
        key = request_key(url, params)
        if self.offline:
            resp = self.store.load(key, self.offline_run)
        else:
            try:
                resp = self._fetch(url, params)
            except ClientError as exc:
                self.store.save(key, url, params, None, error=str(exc))
                raise
            self.store.save(key, url, params, resp)
        if self.latest_fetch is None or resp.fetched_at > self.latest_fetch:
            self.latest_fetch = resp.fetched_at
        return resp

    def _fetch(self, url: str, params: dict[str, Any] | None) -> Response:
        assert self._http is not None
        safe_url = url  # params (which may hold the API key) are never in errors
        last_err: Exception | None = None
        delay = 0.0
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(delay)
            delay = min(2.0 ** (attempt + 1), 30.0)
            try:
                r = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                last_err = ApiError(f"{safe_url}: network error: {exc.__class__.__name__}: {exc}")
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_err = ApiError(f"{safe_url}: HTTP {r.status_code}")
                retry_after = r.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    delay = min(float(retry_after), 60.0)
                continue
            if r.status_code >= 400:
                raise ClientError(f"{safe_url}: HTTP {r.status_code}: {_snippet(r.text)}")
            return Response(body=_parse_json(safe_url, r), headers={
                h: r.headers[h] for h in KEPT_HEADERS if h in r.headers})
        raise last_err or ApiError(f"{safe_url}: failed")


def _snippet(text: str, n: int = 200) -> str:
    return re.sub(r"\s+", " ", text)[:n]


def _parse_json(url: str, r: httpx.Response) -> Any:
    ctype = r.headers.get("content-type", "")
    try:
        return r.json()
    except (json.JSONDecodeError, ValueError):
        kind = "HTML" if "html" in ctype or r.text.lstrip().startswith("<") else "non-JSON"
        raise ApiError(
            f"{url}: expected JSON but got {kind} (content-type {ctype!r}): {_snippet(r.text)}"
        ) from None
