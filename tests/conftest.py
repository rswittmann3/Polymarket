"""Shared fixtures: a mock HTTP transport that serves tests/fixtures/synthetic."""
from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import httpx
import pytest

from pm_edges.config import Config
from pm_edges.pipeline import GameSpec

FIX = Path(__file__).parent / "fixtures" / "synthetic"


def load(name: str):
    return json.loads((FIX / name).read_text())


class FakeApis:
    """Routes Gamma, CLOB and Odds API requests to fixture data. Tests may
    edit `self.data` before running."""

    def __init__(self):
        self.data = {name: load(name) for name in (
            "gamma_event_base.json", "gamma_event_props.json", "gamma_event_quarter.json",
            "gamma_search.json", "clob_books.json", "clob_history.json", "odds_events.json",
            "odds_event_main.json", "odds_event_alt.json")}
        self.requests: list[httpx.Request] = []
        self.quota = 480

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host, path, q = request.url.host, request.url.path, request.url.params
        d = self.data
        if host == "gamma-api.polymarket.com":
            if path == "/events":
                by_slug = {e["slug"]: e for k in ("gamma_event_base.json", "gamma_event_props.json",
                                                   "gamma_event_quarter.json") for e in d[k]}
                ev = by_slug.get(q.get("slug"))
                return httpx.Response(200, json=[ev] if ev else [])
            if path == "/public-search":
                if q.get("q") == "Commanders 49ers":
                    return httpx.Response(200, json=d["gamma_search.json"])
                return httpx.Response(200, json={"events": []})
        if host == "clob.polymarket.com":
            if path == "/book":
                book = d["clob_books.json"].get(q.get("token_id"))
                if book is None:
                    return httpx.Response(404, json={"error": "No orderbook exists for the requested token id"})
                return httpx.Response(200, json=book)
            if path == "/prices-history":
                return httpx.Response(200, json=d["clob_history.json"].get(q.get("market"), {"history": []}))
        if host == "api.the-odds-api.com":
            if not q.get("apiKey"):
                return httpx.Response(401, json={"message": "missing key"})
            if path.endswith("/events"):
                return httpx.Response(200, json=d["odds_events.json"])
            if path.endswith("/odds"):
                markets = q.get("markets", "").split(",")
                body = d["odds_event_alt.json"] if all("alternate" in m for m in markets) else d["odds_event_main.json"]
                cost = len(markets) * len(q.get("regions", "us").split(","))
                self.quota -= cost
                return httpx.Response(200, json=body, headers={
                    "x-requests-remaining": str(self.quota), "x-requests-used": str(500 - self.quota),
                    "x-requests-last": str(cost)})
        return httpx.Response(404, text="<html>not found</html>", headers={"content-type": "text/html"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def fake_apis():
    return FakeApis()


@pytest.fixture
def cfg(tmp_path):
    return Config(data_dir=tmp_path / "data", odds_api_key="test-key", retries=0)


@pytest.fixture
def spec():
    return GameSpec("nfl", "WAS", "SF", date(2026, 10, 19), time(20, 15))
