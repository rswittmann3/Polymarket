"""Offline tests for novig_mm: signing, price grid / touch, quote engine, replay, client and recorder."""
from __future__ import annotations

import base64
import gzip
import json
from decimal import Decimal as D

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
from cryptography.hazmat.primitives.hashes import SHA256

from novig_mm import analyze, recorder, signing
from novig_mm.book import (GRID, depth_within, is_valid_price, ladder, round_down, round_up, step_up, tick_size,
                           touch_from_market)
from novig_mm.client import NovigClient, NovigConfig, NovigError, load_config
from novig_mm.quoter import QuoteConfig, SimOrder, apply_trade, make_quotes

A, B = "aaaaaaaa-0000-0000-0000-000000000001", "bbbbbbbb-0000-0000-0000-000000000002"


# ---------------------------------------------------------------- signing
def test_docs_example_string_to_sign():
    assert signing.string_to_sign(1755000000000, "GET", "/v3/keys") == (
        "NOVIG-V3\n1755000000000\nGET\n/v3/keys\n\n"
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert signing.EMPTY_SHA256.startswith("e3b0c442") and signing.EMPTY_SHA256.endswith("7852b855")


def test_body_hash_differs_for_empty_object():
    assert signing.string_to_sign(1, "POST", "/v3/echo", "", b"{}") != signing.string_to_sign(1, "POST", "/v3/echo", "", b"")


@pytest.mark.parametrize("raw,want", [
    ("", ""),
    ("b=2&a=1", "a=1&b=2"),
    ("a=2&a=1", "a=1&a=2"),                              # repeated pairs kept, sorted by value
    ("flag", "flag="),                                    # no '=' -> empty value
    ("q=a+b", "q=a%2Bb"),                                 # bare '+' stays a literal plus, re-encoded
    ("q=a%20b", "q=a%20b"),
    ("q=%e2%82%ac", "q=%E2%82%AC"),                       # UTF-8 decoded then uppercase re-encode
    ("n=a~b-c.d_e", "n=a~b-c.d_e"),                       # unreserved characters stay
    ("x=1=2", "x=1%3D2"),                                 # split at the first '='
])
def test_canonical_query(raw, want):
    assert signing.canonical_query(raw) == want


def _pem(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def test_ed25519_sign_verifies():
    k = ed25519.Ed25519PrivateKey.generate()
    key = signing.load_private_key(_pem(k))
    h = signing.signed_headers("kid", key, "GET", "/v3/keys", "b=2&a=1", b"", now_ms=1755000000000)
    assert set(h) == {"Novig-Key-Id", "Novig-Timestamp", "Novig-Signature"} and h["Novig-Timestamp"] == "1755000000000"
    text = signing.string_to_sign(1755000000000, "GET", "/v3/keys", "b=2&a=1")
    k.public_key().verify(base64.b64decode(h["Novig-Signature"]), text.encode())     # raises if wrong


def test_p256_signature_is_der_and_verifies():
    k = ec.generate_private_key(ec.SECP256R1())
    key = signing.load_private_key(_pem(k))
    sig = base64.b64decode(signing.sign_text(key, "hello"))
    assert 68 <= len(sig) <= 72 and sig[0] == 0x30                                     # DER SEQUENCE
    k.public_key().verify(sig, b"hello", ECDSA(SHA256()))


def test_rejects_other_curves():
    with pytest.raises(ValueError):
        signing.load_private_key(_pem(ec.generate_private_key(ec.SECP384R1())))


# ---------------------------------------------------------------- grid / touch
def test_grid_shape_and_validity():
    assert GRID[0] == D("0.001") and GRID[-1] == D("0.999") and list(GRID) == sorted(set(GRID))
    assert is_valid_price(D("0.050")) and is_valid_price(D("0.055")) and is_valid_price(D("0.945")) and is_valid_price(D("0.950"))
    assert is_valid_price(D("0.665")) and not is_valid_price(D("0.667")) and not is_valid_price(D("0.052"))   # docs: 0.667 is off-grid
    assert not is_valid_price(D("0.9455"))


def test_rounding_and_ticks():
    assert round_down(D("0.5249")) == D("0.520") and round_down(D("0.525")) == D("0.525")
    assert round_up(D("0.521")) == D("0.525")
    assert round_down(D("0.0004")) is None and round_up(D("0.9995")) is None
    assert step_up(D("0.945")) == D("0.950") and step_up(D("0.050")) == D("0.055") and step_up(D("0.500"), -1) == D("0.495")
    assert tick_size(D("0.5")) == D("0.005") and tick_size(D("0.02")) == D("0.001") and tick_size(D("0.97")) == D("0.001")


def mkt(bid_a=("0.45", 10000), bid_b=("0.44", 10000), status="OPEN", starts=0):
    def o(oid, name, b):
        d = {"outcomeId": oid, "name": name, "status": "TBD"}
        if b:
            d["bid"] = {"price": b[0], "qty": b[1]}
        return d
    return {"marketId": "m1", "eventId": "e1", "description": "X @ Y", "marketType": "MONEY", "status": status,
            "startsTs": starts, "fee": {"coefficient": "0.06", "makerCredit": "0.25", "charged": "WHEN_LIVE"},
            "volume": 5, "openInterest": 7, "outcomes": [o(A, "X", bid_a), o(B, "Y", bid_b)]}


def test_touch_spread_mid_ask():
    t = touch_from_market(mkt())
    assert t.spread == D("0.11") and not t.crossed
    assert t.ask(0) == D("0.56") and t.ask(1) == D("0.55") and t.ask_qty(0) == 10000
    assert t.mid(0) == D("0.505") and t.mid(1) == D("0.495")


def test_touch_one_sided_and_non_binary():
    t = touch_from_market(mkt(bid_b=None))
    assert t.spread is None and t.mid(0) is None and t.ask(0) is None and t.ask(1) == D("0.55")
    m = mkt(); m["outcomes"].append({"outcomeId": "c", "name": "Draw", "status": "TBD"})
    assert touch_from_market(m) is None


def test_ladder_and_depth():
    orders = [{"order": "1", "price": "0.665", "qty": 110}, {"order": "2", "price": "0.665", "qty": 40},
              {"order": "3", "price": "0.660", "qty": 250}, {"order": "4", "price": "0.600", "qty": 9}]
    lv = ladder(orders)
    assert [(l.price, l.qty, l.orders) for l in lv] == [(D("0.665"), 150, 2), (D("0.660"), 250, 1), (D("0.600"), 9, 1)]
    assert depth_within(lv, D("0.665"), 0) == 150 and depth_within(lv, D("0.665"), 1) == 400
    assert depth_within(lv, None, 3) == 0


# ---------------------------------------------------------------- quotes
def test_basic_quotes_symmetric_and_on_grid():
    t = touch_from_market(mkt(bid_a=("0.48", 500), bid_b=("0.48", 500)))
    qs = make_quotes(t, D("0.50"), D("0.01"), 0, 120, QuoteConfig(half_spread=D("0.015"), size=100))
    assert qs.skipped is None and len(qs.quotes) == 2
    by = {q.outcome_id: q for q in qs.quotes}
    assert by[A].price == D("0.485") and by[B].price == D("0.485")        # 0.50 - 0.015 rounded down to the grid
    assert all(is_valid_price(q.price) and q.tif == "PO" and q.qty == 100 for q in qs.quotes)
    assert by[A].ticks_from_best == 1                                      # one tick above the 0.48 touch


def test_never_crosses_the_opposing_best_price():
    # B is bid at 0.60, so buying A at >= 0.40 would take. Fair says A=0.50 (disagreement is allowed by a wide cap)
    t = touch_from_market(mkt(bid_a=("0.30", 100), bid_b=("0.60", 100)))
    qs = make_quotes(t, D("0.50"), D("0.01"), 0, 120, QuoteConfig(max_book_disagreement=D("0.5")))
    a = next(q for q in qs.quotes if q.outcome_id == A)
    assert a.price + D("0.60") < 1 and a.price == D("0.395")


def test_inventory_skew_and_cap():
    t = touch_from_market(mkt(bid_a=("0.48", 500), bid_b=("0.48", 500)))
    cfg = QuoteConfig(max_inventory=300)
    flat = {q.outcome_id: q.price for q in make_quotes(t, D("0.50"), D("0.01"), 0, 120, cfg).quotes}
    long_a = {q.outcome_id: q.price for q in make_quotes(t, D("0.50"), D("0.01"), 200, 120, cfg).quotes}
    assert long_a[A] < flat[A] and long_a[B] > flat[B]                    # lean away from more A, toward selling A
    capped = make_quotes(t, D("0.50"), D("0.01"), 300, 120, cfg)
    assert [q.outcome_id for q in capped.quotes] == [B]                   # at the cap: no more buying A
    short = make_quotes(t, D("0.50"), D("0.01"), -300, 120, cfg)
    assert [q.outcome_id for q in short.quotes] == [A]


@pytest.mark.parametrize("kw,fragment", [
    (dict(minutes_to_start=30), "inside"),
    (dict(minutes_to_start=120, fair=D("0.02")), "outside quoting range"),
    (dict(minutes_to_start=120, fair=D("0.62")), "disagreement"),
    (dict(minutes_to_start=120, market_open=False), "not open"),
])
def test_stand_down(kw, fragment):
    t = touch_from_market(mkt(bid_a=("0.48", 500), bid_b=("0.48", 500)))
    args = dict(fair=D("0.50"), minutes_to_start=120); args.update(kw)
    qs = make_quotes(t, args["fair"], D("0.01"), 0, args["minutes_to_start"], QuoteConfig(), market_open=args.get("market_open", True))
    assert qs.quotes == () and fragment in qs.skipped


def test_sd_widens_the_quote():
    t = touch_from_market(mkt(bid_a=("0.48", 500), bid_b=("0.48", 500)))
    tight = make_quotes(t, D("0.50"), D("0.005"), 0, 120).quotes[0].price
    wide = make_quotes(t, D("0.50"), D("0.04"), 0, 120).quotes[0].price
    assert wide < tight


# ---------------------------------------------------------------- replay
def test_replay_price_priority_and_queue():
    o = SimOrder(A, D("0.500"), 100, queue_ahead=50)
    assert apply_trade(o, B, D("0.400"), 999) == 0                        # wrong outcome
    assert apply_trade(o, A, D("0.510"), 999) == 0                        # trade at a better price than ours: not our turn
    assert apply_trade(o, A, D("0.500"), 60) == 10                        # 60 traded at our price, 50 ahead of us
    assert apply_trade(o, A, D("0.500"), 60) == 60                        # next 60 are all ours (no double counting)
    assert apply_trade(o, A, D("0.500"), 60) == 30 and o.remaining == 0   # only 30 left to fill
    full = SimOrder(A, D("0.500"), 100, queue_ahead=1000)
    assert apply_trade(full, A, D("0.495"), 1) == 100                     # trade through our price fills us completely


# ---------------------------------------------------------------- client
def client(handler, **kw):
    return NovigClient(NovigConfig(**kw), http=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None,
                       read_rate_hz=1000.0)


def test_no_trading_methods():
    banned = ("place", "cancel", "order", "transfer", "fund", "withdraw")
    assert not [n for n in dir(NovigClient) if not n.startswith("_") and any(b in n.lower() for b in banned)]


def test_public_paths_only_and_pagination():
    seen = []

    def h(req: httpx.Request):
        seen.append(req)
        after = req.url.params.get("after")
        if after is None:
            return httpx.Response(200, json={"items": [{"i": 1}], "next": "c1"})
        return httpx.Response(200, json={"items": [{"i": 2}]})

    c = client(h)
    with pytest.raises(ValueError):
        c.get_public("/v3/catalog/markets")
    assert [x["i"] for x in c.pages("/v3/public/catalog/events", {"league": "NBA"})] == [1, 2]
    assert "Novig-Signature" not in seen[0].headers                        # public calls are unsigned
    assert seen[1].url.params["after"] == "c1" and seen[0].url.params["league"] == "NBA"


def test_retry_on_429_then_error_parsing():
    calls = {"n": 0}
    sleeps = []

    def h(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"code": "RATE_LIMIT_EXCEEDED"})
        return httpx.Response(200, json={"ok": True})

    c = NovigClient(NovigConfig(), http=httpx.Client(transport=httpx.MockTransport(h)), sleep=sleeps.append, read_rate_hz=1000.0)
    assert c.get_public("/v3/public/catalog/events") == {"ok": True} and 2.0 in sleeps

    c2 = client(lambda r: httpx.Response(409, json={"code": "MARKET_CLOSED", "message": "closed"}))
    with pytest.raises(NovigError) as e:
        c2.get_public("/v3/public/catalog/events")
    assert e.value.status == 409 and e.value.code == "MARKET_CLOSED"
    c3 = client(lambda r: httpx.Response(403, text="<html>edge</html>", headers={"content-type": "text/html"}))
    with pytest.raises(NovigError) as e:
        c3.get_public("/v3/public/catalog/events")
    assert e.value.status == 403 and e.value.code is None


def test_signed_request_headers_and_echo():
    k = ed25519.Ed25519PrivateKey.generate()
    seen = []

    def h(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, content=req.content)

    c = client(h, key_id="kid-1", pem=_pem(k))
    assert c.echo({"a": 1}) == {"a": 1}
    r = seen[0]
    assert r.method == "POST" and r.url.path == "/v3/echo" and r.headers["content-type"] == "application/json"
    text = signing.string_to_sign(r.headers["Novig-Timestamp"], "POST", "/v3/echo", "", r.content)
    k.public_key().verify(base64.b64decode(r.headers["Novig-Signature"]), text.encode())
    with pytest.raises(NovigError):
        client(h).echo()                                                   # no key configured


def test_config_loading_and_redaction(tmp_path):
    pem = tmp_path / "k.pem"
    pem.write_bytes(_pem(ed25519.Ed25519PrivateKey.generate()))
    cfg = load_config({"NOVIG_HOST": "https://api.paper.novig.com/", "NOVIG_KEY_ID": "kid", "NOVIG_PEM": str(pem)})
    assert cfg.host == "https://api.paper.novig.com" and cfg.key_id == "kid" and b"BEGIN" in cfg.pem
    assert "BEGIN" not in repr(cfg) and "kid" not in repr(cfg)
    assert load_config({}).host == "https://api.novig.com" and load_config({}).pem is None


# ---------------------------------------------------------------- recorder / analyze
def test_recorder_roundtrip_and_summary(tmp_path):
    ts = 1_793_840_000_000
    start = ts + 3 * 3600_000
    m = mkt(starts=start)
    closed = mkt(status="CLOSED", starts=start)
    nonbin = mkt(starts=start); nonbin["outcomes"].append({"outcomeId": "c", "name": "Draw", "status": "TBD"})
    rows = recorder.rows_from_markets([m, closed, nonbin], ts, "NBA")
    assert len(rows) == 1 and rows[0]["b"] == ["0.45", "0.44"] and rows[0]["ch"] == "WHEN_LIVE"

    path = recorder.out_path(tmp_path, ts)
    recorder.append_rows(path, rows)
    recorder.append_rows(path, [dict(rows[0], ts=ts + 30_000, b=["0.46", None])])         # a second gzip member
    got = list(analyze.read_rows([path]))
    assert len(got) == 2
    s = analyze.summarize(got)
    assert len(s) == 1 and s[0]["league"] == "NBA" and s[0]["to_start"] == "120-360m" and s[0]["line"] == "main"
    assert s[0]["vol_med"] == 5
    assert s[0]["obs"] == 2 and s[0]["two_sided_%"] == 50.0 and s[0]["spread_c_med"] == 11.0 and s[0]["touch_qty_med"] == 10000


def _pair(ts, taker_out, price, qty, market="m1", other={"a": "b", "b": "a"}):
    return [
        {"timestamp": ts, "outcomeId": taker_out, "marketId": market, "tradeType": "STRAIGHT", "cost": price * qty, "qty": qty, "side": "TAKER"},
        {"timestamp": ts, "outcomeId": other[taker_out], "marketId": market, "tradeType": "STRAIGHT", "cost": (1 - price) * qty, "qty": qty, "side": "MAKER"},
    ]


def test_flow_markout_phase_and_sign():
    pd = pytest.importorskip("pandas")
    from novig_mm import flow
    start_ms = 1_793_700_000_000                                      # 2026-11-03T10:00:00Z
    meta = pd.DataFrame([{"marketId": "m1", "lg": "NBA", "mt": "MONEY", "st": start_ms, "ch": "WHEN_LIVE", "line": "main"}]).set_index("marketId")
    rows = []
    rows += _pair("2026-11-03T08:00:00.000Z", "a", 0.50, 100)         # taker buys A at .50 two hours before the start ...
    rows += _pair("2026-11-03T08:01:30.000Z", "a", 0.55, 100)         # ... and A is .55 ninety seconds later: taker was right
    rows += _pair("2026-11-03T08:05:00.000Z", "b", 0.45, 100)         # taker buys B at .45 == A at .55 (dir -1)
    rows += _pair("2026-11-03T08:06:30.000Z", "b", 0.45, 100)         # price unchanged -> zero markout for the previous trade
    rows += _pair("2026-11-03T10:30:00.000Z", "a", 0.60, 50)          # after the scheduled start: live
    raw = pd.DataFrame(rows)
    # canonical outcome is the smaller id, 'a'
    m = flow.trade_markouts(raw, meta, horizons=((60, 60),))
    first = m.iloc[0]
    assert first.phase == "pre 1-6h" and abs(first.mins_to_start - 120) < 1e-9
    assert first["maker_60s"] == pytest.approx(-5.0)                  # takers gained 5c, so makers lost 5c
    third = m.iloc[2]
    assert third.dir == -1 and third.pc == pytest.approx(0.55) and third["maker_60s"] == pytest.approx(0.0)
    assert m.iloc[-1].phase == "live" and m.iloc[-1].mins_to_start < 0
    assert m.iloc[-1]["maker_60s"] != m.iloc[-1]["maker_60s"]         # NaN: no later trades to measure against


def test_flow_summary_has_live_credit():
    pd = pytest.importorskip("pandas")
    from novig_mm import flow
    start_ms = 1_793_700_000_000
    meta = pd.DataFrame([{"marketId": "m1", "lg": "NBA", "mt": "MONEY", "st": start_ms, "ch": "WHEN_LIVE", "line": "main"}]).set_index("marketId")
    rows = []
    for i in range(80):                                               # 80 live prints, 1 per 10 s, price flat at .50
        rows += _pair(f"2026-11-03T10:{i // 6:02d}:{(i % 6) * 10:02d}.000Z", "a", 0.50, 100)
    s = flow.summarize(flow.trade_markouts(pd.DataFrame(rows), meta, horizons=((60, 60), (300, 300))), "maker_60s")
    assert len(s) == 1 and s.iloc[0].phase == "live" and s.iloc[0]["maker_c"] == pytest.approx(0.0)
    assert s.iloc[0]["live_credit_$"] == pytest.approx(0.015 * 0.25 * 100 * 80)


def test_main_line_is_the_strike_closest_to_even():
    def row(m, b0, b1, e="e1"):
        return {"ts": 0, "m": m, "e": e, "lg": "NBA", "mt": "SPREAD", "st": 600000, "vol": 1, "b": [b0, b1], "q": [10, 10]}
    rows = [row("alt-lo", "0.20", "0.70"), row("main", "0.49", "0.49"), row("alt-hi", "0.80", "0.10"),
            row("one-sided", "0.50", None), row("other-event", "0.30", "0.60", e="e2")]
    assert analyze.main_lines(rows) == {"main", "other-event"}


def test_poll_once_uses_window_and_types(tmp_path):
    seen = []

    def h(req: httpx.Request):
        seen.append(dict(req.url.params))
        return httpx.Response(200, json={"items": [mkt(starts=1_793_850_000_000)]})

    c = client(h)
    n = recorder.poll_once(c, ["NBA", "NFL"], "MONEY,SPREAD", 12, tmp_path, now_ms=1_793_840_000_000)
    assert n == 2 and seen[0]["league"] == "NBA" and seen[1]["league"] == "NFL"
    assert seen[0]["marketType"] == "MONEY,SPREAD" and seen[0]["eventStatus"] == "OPEN_PREGAME"
    assert int(seen[0]["startsBefore"]) == 1_793_840_000_000 + 12 * 3600_000
    assert list(tmp_path.glob("touch_*.jsonl.gz"))
