import json

import httpx
import pytest

from pm_edges.httpclient import ApiError, JsonClient, OfflineMiss, RawStore, request_key


def client(tmp_path, handler, **kw):
    store = RawStore(tmp_path, "g", run_id="r1")
    return JsonClient(store, transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)


def test_retries_then_succeeds(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json={"ok": True})
    c = client(tmp_path, handler, retries=3)
    assert c.get("https://gamma-api.polymarket.com/events", {"slug": "x"}).body == {"ok": True}
    assert len(calls) == 3


def test_gives_up_after_retries(tmp_path):
    c = client(tmp_path, lambda req: httpx.Response(500), retries=2)
    with pytest.raises(ApiError, match="HTTP 500"):
        c.get("https://gamma-api.polymarket.com/events")


def test_html_instead_of_json_is_a_clear_error(tmp_path):
    def handler(req):
        return httpx.Response(200, text="<!doctype html><html>Cloudflare</html>",
                              headers={"content-type": "text/html"})
    with pytest.raises(ApiError, match="expected JSON but got HTML"):
        client(tmp_path, handler, retries=0).get("https://clob.polymarket.com/book", {"token_id": "1"})


def test_4xx_not_retried(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(404, json={"error": "nope"})
    with pytest.raises(ApiError, match="404"):
        client(tmp_path, handler, retries=3).get("https://clob.polymarket.com/book")
    assert len(calls) == 1


def test_api_key_never_saved_and_offline_replay(tmp_path):
    def handler(req):
        return httpx.Response(200, json=[{"id": 1}], headers={"x-requests-remaining": "499"})
    c = client(tmp_path, handler)
    url = "https://api.the-odds-api.com/v4/sports/x/events"
    c.get(url, {"apiKey": "SECRET123", "dateFormat": "iso"})
    saved = list((tmp_path / "g" / "r1").glob("*.json"))
    assert len(saved) == 1
    text = saved[0].read_text()
    assert "SECRET123" not in text
    assert json.loads(text)["headers"]["x-requests-remaining"] == "499"

    off = JsonClient(RawStore(tmp_path, "g"), offline=True)
    resp = off.get(url, {"apiKey": "different", "dateFormat": "iso"})
    assert resp.body == [{"id": 1}] and resp.from_cache
    with pytest.raises(OfflineMiss):
        off.get(url, {"dateFormat": "other"})


def test_request_key_ignores_secret_and_param_order():
    a = request_key("https://x.com/a/b", {"apiKey": "1", "m": "h2h", "r": "us"})
    b = request_key("https://x.com/a/b", {"r": "us", "m": "h2h", "apiKey": "2"})
    assert a == b
