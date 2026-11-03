"""NOVIG-V3 request signing (https://docs.novig.com/api/signing.md).

Every signed request carries three headers:

    Novig-Key-Id     the key's UUID (not part of the signed string)
    Novig-Timestamp  Unix milliseconds, unpadded decimal, within +-30 s of the server
    Novig-Signature  standard padded base64 of the signature

The signed string is six lines joined by LF, no trailing newline:

    NOVIG-V3
    {unix_millis}
    {METHOD}
    {path}                (leading slash, not normalised or percent-decoded)
    {canonical_query}     (empty line when there is no query)
    {lowercase_hex(sha256(raw_body))}

Ed25519 signs that string directly. P-256 is ECDSA over SHA-256 with a DER signature
(which is what `cryptography` returns; WebCrypto's raw r||s never verifies).
"""
from __future__ import annotations

import base64
import hashlib
import time
from typing import Union
from urllib.parse import quote, unquote

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
from cryptography.hazmat.primitives.hashes import SHA256

PrivateKey = Union[ed25519.Ed25519PrivateKey, ec.EllipticCurvePrivateKey]

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def canonical_query(raw_query: str) -> str:
    """Docs algorithm: split on '&', split each pair at the first '=', percent-decode both parts
    (a bare '+' stays '+'), re-encode keeping A-Za-z0-9 - . _ ~ with uppercase %XX for the rest,
    sort by name then value (repeated pairs kept), join with '&'. A pair without '=' has an empty value."""
    pairs = []
    for part in raw_query.split("&"):
        if not part:
            continue
        name, _, value = part.partition("=")
        pairs.append((quote(unquote(name), safe=""), quote(unquote(value), safe="")))
    pairs.sort()
    return "&".join(f"{n}={v}" for n, v in pairs)


def string_to_sign(timestamp_ms: int | str, method: str, path: str, raw_query: str = "", body: bytes = b"") -> str:
    return "\n".join([
        "NOVIG-V3",
        str(timestamp_ms),
        method.upper(),
        path,
        canonical_query(raw_query),
        hashlib.sha256(body).hexdigest(),
    ])


def load_private_key(pem: bytes | str) -> PrivateKey:
    """Load an unencrypted PKCS#8 PEM (the format `openssl genpkey` writes) for Ed25519 or P-256."""
    data = pem.encode() if isinstance(pem, str) else pem
    key = serialization.load_pem_private_key(data, password=None)
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key
    if isinstance(key, ec.EllipticCurvePrivateKey) and key.curve.name == "secp256r1":
        return key
    raise ValueError("Novig keys must be Ed25519 or P-256 (prime256v1)")


def sign_text(key: PrivateKey, text: str) -> str:
    raw = text.encode()
    if isinstance(key, ed25519.Ed25519PrivateKey):
        sig = key.sign(raw)
    else:
        sig = key.sign(raw, ECDSA(SHA256()))
    return base64.b64encode(sig).decode()


def signed_headers(key_id: str, key: PrivateKey, method: str, path: str, raw_query: str = "",
                   body: bytes = b"", now_ms: int | None = None) -> dict[str, str]:
    ts = int(time.time() * 1000) if now_ms is None else now_ms
    text = string_to_sign(ts, method, path, raw_query, body)
    return {"Novig-Key-Id": key_id, "Novig-Timestamp": str(ts), "Novig-Signature": sign_text(key, text)}
