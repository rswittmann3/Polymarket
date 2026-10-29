"""Polymarket US (CFTC-regulated exchange) client: public book reads and Ed25519-signed trading calls.

Separate system from polymarket.com's CLOB: https://docs.polymarket.us/api-reference/introduction
  * public data   https://gateway.polymarket.us  (no key)
  * trading       https://api.polymarket.us      (X-PM-Access-Key / X-PM-Timestamp / X-PM-Signature)

Each market is ONE order book on the long ("Yes"/Up) side. Buying the short side (Down) at q is the
same as selling the long at 1 - q, so Up-ask + Down-ask = 1 + spread and can never be below 1.
"""
from __future__ import annotations

import base64
import datetime as dt
import time
from dataclasses import dataclass

import httpx

GATEWAY = "https://gateway.polymarket.us"
API = "https://api.polymarket.us"
FEE_THETA = 0.0695  # standard taker coefficient; fee = theta * contracts * p * (1 - p)


def taker_fee(price: float, qty: float, theta: float = FEE_THETA) -> float:
    return theta * qty * price * (1 - price)


def updown_slug(start_ts: int, asset: str = "btc", horizon: str = "5m") -> str:
    return f"cpc-{asset}-updown-{horizon}-{dt.datetime.fromtimestamp(start_ts, dt.UTC):%Y-%m-%d-%H%Mz}"


@dataclass(frozen=True)
class Book:
    slug: str
    state: str
    bids: list[tuple[float, float]]    # best first
    offers: list[tuple[float, float]]  # best first

    @property
    def open(self) -> bool:
        return self.state == "MARKET_STATE_OPEN"


def get_book(client: httpx.Client, slug: str) -> Book | None:
    r = client.get(f"{GATEWAY}/v1/markets/{slug}/book")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    d = r.json()["marketData"]
    lv = lambda rows: [(float(x["px"]["value"]), float(x["qty"])) for x in rows]  # noqa: E731
    return Book(slug, d.get("state", ""), sorted(lv(d.get("bids", [])), reverse=True),
                sorted(lv(d.get("offers", []))))


class Signer:
    """secret_key is the base64 string from polymarket.us/developer; the first 32 bytes are the seed."""

    def __init__(self, key_id: str, secret_key: str) -> None:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        self.key_id = key_id
        self._key = ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(secret_key)[:32])

    def headers(self, method: str, path: str, now_ms: int | None = None) -> dict[str, str]:
        ts = str(now_ms if now_ms is not None else int(time.time() * 1000))
        sig = base64.b64encode(self._key.sign(f"{ts}{method}{path}".encode())).decode()
        return {"X-PM-Access-Key": self.key_id, "X-PM-Timestamp": ts, "X-PM-Signature": sig,
                "Content-Type": "application/json"}


class Trader:
    def __init__(self, signer: Signer, client: httpx.Client | None = None) -> None:
        self.s = signer
        self.c = client or httpx.Client(timeout=15.0)

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        r = self.c.request(method, API + path, headers=self.s.headers(method, path), json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else {}

    def balances(self) -> dict:
        return self._call("GET", "/v1/account/balances")

    def preview(self, order: dict) -> dict:
        return self._call("POST", "/v1/order/preview", {"request": order})  # preview wraps the order

    def create(self, order: dict) -> dict:
        return self._call("POST", "/v1/orders", order)

    def close_position(self, slug: str) -> dict:
        return self._call("POST", "/v1/order/close-position", {"marketSlug": slug})


def limit_order(slug: str, intent: str, price: float, qty: float, tif: str) -> dict:
    return {"marketSlug": slug, "intent": intent, "type": "ORDER_TYPE_LIMIT",
            "price": {"value": f"{price:.2f}", "currency": "USD"}, "quantity": qty, "tif": tif,
            "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
            "synchronousExecution": True, "maxBlockTime": "5"}
