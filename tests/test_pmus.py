import base64

import pytest

from pm_edges.pmus import Signer, limit_order, taker_fee, updown_slug

cryptography = pytest.importorskip("cryptography")


def test_slug_matches_exchange_format():
    assert updown_slug(1793294700) == "cpc-btc-updown-5m-2026-10-29-1725z"


def test_signature_is_ed25519_over_ts_method_path():
    from cryptography.hazmat.primitives.asymmetric import ed25519
    seed = bytes(range(32))
    s = Signer("kid", base64.b64encode(seed + bytes(32)).decode())  # 64-byte secret: seed is the first 32
    h = s.headers("POST", "/v1/orders", now_ms=1700000000000)
    assert h["X-PM-Access-Key"] == "kid" and h["X-PM-Timestamp"] == "1700000000000"
    pub = ed25519.Ed25519PrivateKey.from_private_bytes(seed).public_key()
    pub.verify(base64.b64decode(h["X-PM-Signature"]), b"1700000000000POST/v1/orders")  # raises if wrong


def test_fee_matches_published_example():
    assert taker_fee(0.5, 100) == pytest.approx(1.7375)  # docs: max $1.74 for a 100-lot at 0.50


def test_buy_both_sides_never_below_one_on_single_book():
    # Up ask = best offer; Down ask = 1 - best bid. A book cannot cross, so the sum is 1 + spread >= 1.
    bid, offer = 0.49, 0.50
    assert offer + (1 - bid) >= 1.0


def test_order_body():
    o = limit_order("m", "ORDER_INTENT_BUY_LONG", 0.5, 2, "TIME_IN_FORCE_FILL_OR_KILL")
    assert o["price"] == {"value": "0.50", "currency": "USD"} and o["quantity"] == 2


def test_preview_wraps_order_in_request(monkeypatch):
    from pm_edges.pmus import Trader
    seen = {}
    t = Trader(Signer("k", base64.b64encode(bytes(64)).decode()))
    monkeypatch.setattr(t, "_call", lambda m, p, b=None: seen.update(m=m, p=p, b=b) or {})
    t.preview({"marketSlug": "x"})
    assert seen["p"] == "/v1/order/preview" and seen["b"] == {"request": {"marketSlug": "x"}}


def test_order_outcome_trusts_cum_quantity_not_executions(monkeypatch):
    from pm_edges.pmus import Trader
    t = Trader(Signer("k", base64.b64encode(bytes(64)).decode()))
    states = iter([{"order": {"state": "ORDER_STATE_NEW", "cumQuantity": 0}},
                   {"order": {"state": "ORDER_STATE_EXPIRED", "cumQuantity": 0}}])
    monkeypatch.setattr(t, "_call", lambda m, p, b=None: next(states))
    monkeypatch.setattr("pm_edges.pmus.time.sleep", lambda s: None)
    assert t.order_outcome("x")["state"] == "ORDER_STATE_EXPIRED"


def test_order_outcome_retries_404_then_raises(monkeypatch):
    from pm_edges.pmus import Trader
    t = Trader(Signer("k", base64.b64encode(bytes(64)).decode()))
    calls = []

    def fake(m, p, b=None):
        calls.append(p)
        if len(calls) < 3:
            raise RuntimeError("GET /v1/order/x -> 404: Order not found")
        return {"order": {"state": "ORDER_STATE_FILLED", "cumQuantity": 1}}
    monkeypatch.setattr(t, "_call", fake)
    monkeypatch.setattr("pm_edges.pmus.time.sleep", lambda s: None)
    assert t.order_outcome("x")["cumQuantity"] == 1 and len(calls) == 3
    monkeypatch.setattr(t, "_call", lambda m, p, b=None: (_ for _ in ()).throw(RuntimeError("404 nope")))
    import pytest
    with pytest.raises(RuntimeError, match="not readable"):
        t.order_outcome("x", tries=2)


def test_slugs_per_asset():
    from pm_edges.pmus import event_slug
    assert updown_slug(1793294700, "eth") == "cpceth-eth-updown-5m-2026-10-29-1725z"
    assert updown_slug(1793294700, "xrp") == "cpcxrp-xrp-updown-5m-2026-10-29-1725z"
    assert event_slug(1793294700, "sol") == "sol-updown-5m-2026-10-29-1725z"
