"""Credential redaction.

Its own module, with no dependencies beyond the standard library, for one
reason: **everything that can emit text needs it**, and the logger must not have
to import the video decoder — and therefore OpenCV — in order to be safe.

`decode.py` re-exports these, so the public spelling stays
`from sentinel.decode import redact_url`. This module is the implementation and
the single definition of what a credential looks like; there must never be a
second one, because two redactors drift and the one that drifts is the one that
leaks.
"""

from __future__ import annotations

#: Query parameters whose value is a credential. Cameras and NVRs routinely put
#: one here instead of in the userinfo, and a redactor that only strips userinfo
#: passes the password through untouched while looking like it worked.
CREDENTIAL_QUERY_KEYS = frozenset(
    {
        "password", "passwd", "pwd", "pass", "secret", "token", "auth",
        "key", "apikey", "api_key", "access_token", "accesstoken",
        "signature", "sig", "credential", "credentials", "session",
    }
)

#: Substituted for every removed secret. A fixed marker, never one character per
#: character: the length of a password is information an attacker can use.
REDACTED = "***"


def redact_url(url: str) -> str:
    """Strip every credential from a URL so it is safe to log or display.

    ``rtsp://admin:hunter2@10.0.0.5/stream`` becomes
    ``rtsp://admin:***@10.0.0.5/stream``. The username survives because
    operators identify cameras by it and it is not a secret; the password never
    appears in any form, including its length.

    Written defensively, because this function is the only thing standing between
    a camera password and every log line, error message and database row in the
    system. Three ways an earlier version leaked, all now covered:

    - **It parsed before it redacted.** ``urlsplit`` succeeds but ``.port``
      raises ``ValueError`` on a non-numeric port, and ``.hostname`` returns
      ``None`` for shapes it does not recognise. Both paths returned the input
      verbatim. Nothing here depends on a successful parse: the userinfo is
      removed by string surgery on the netloc, which cannot fail.
    - **It only ever looked at the userinfo.** A credential in the query string
      survived untouched.
    - **It rebuilt the host from ``.hostname``**, which strips the brackets from
      an IPv6 literal and produced an unparseable display URL.

    When anything is uncertain the function fails *closed* — it returns a marker
    rather than the input, because echoing a string that might contain a
    password is the one outcome that must never happen.
    """
    if not isinstance(url, str) or not url:
        return "<no source>"

    # A local path is not a URL and has no credential to strip. Recognised
    # before any parsing, so a Windows path like C:\media\clip.mp4 is never
    # mangled by scheme detection.
    if "://" not in url:
        # A path, not a URL — returned intact so an operator can find the file.
        # The query pass still runs: "?" is illegal in a Windows filename and
        # vanishingly rare in a POSIX one, so redacting a credential-shaped
        # parameter here costs nothing and covers a schemeless "host/s?token=x".
        return _redact_query(url if "@" not in url else "<redacted path>")

    scheme, _, remainder = url.partition("://")
    netloc, slash, tail = remainder.partition("/")

    # Userinfo removal by string surgery. rpartition, not partition: a password
    # may itself contain an "@", and only the last one separates host from
    # userinfo.
    if "@" in netloc:
        userinfo, _, host = netloc.rpartition("@")
        username = userinfo.split(":", 1)[0]
        netloc = (f"{username}:{REDACTED}@" if username else f"{REDACTED}@") + host

    rebuilt = f"{scheme}://{netloc}"
    if slash:
        rebuilt += "/" + tail

    return _redact_query(rebuilt)


def _redact_query(url: str) -> str:
    """Replace credential-shaped query values, leaving the rest legible."""
    head, sep, query = url.partition("?")
    if not sep or not query:
        return url

    query, hash_sep, fragment = query.partition("#")

    redacted = []
    for pair in query.split("&"):
        name, has_value, _ = pair.partition("=")
        if has_value and name.lower() in CREDENTIAL_QUERY_KEYS:
            redacted.append(f"{name}={REDACTED}")
        else:
            redacted.append(pair)

    return head + "?" + "&".join(redacted) + (hash_sep + fragment if hash_sep else "")


def redact_text(text: str) -> str:
    """Redact every credential-bearing URL inside an arbitrary string.

    The backstop for log records. `redact_url` is the primary defence and every
    message *should* already be built from a redacted value — but a log line is
    assembled from whatever a caller passed, including an exception raised by a
    library that has no idea what a camera password is. This finds URL-shaped
    substrings in free text and redacts each one.

    Deliberately narrow. It looks for ``scheme://`` and stops at whitespace or a
    quote, because a redactor that tried to be clever about free text would
    mangle ordinary messages and get switched off.
    """
    if not text or "://" not in text:
        return text

    out: list[str] = []
    rest = text

    while "://" in rest:
        head, sep, tail = rest.partition("://")
        # Walk back to the start of the scheme: the run of scheme characters
        # immediately before "://".
        cut = len(head)
        while cut > 0 and (head[cut - 1].isalnum() or head[cut - 1] in "+-."):
            cut -= 1

        out.append(head[:cut])
        scheme = head[cut:]

        # The URL ends at the first character that cannot be in one.
        end = len(tail)
        for index, character in enumerate(tail):
            if character.isspace() or character in "\"'`<>,;)]}":
                end = index
                break

        out.append(redact_url(scheme + "://" + tail[:end]))
        rest = tail[end:]

    out.append(rest)
    return "".join(out)


def contains_credential(text: str, url: str) -> bool:
    """Whether ``text`` leaks any secret held in ``url``.

    Used by the tests rather than by the runtime. It exists so a test can assert
    the absence of *this URL's* secrets rather than of one hard-coded sentinel,
    which is what lets it catch a leak through a path nobody thought of.
    """
    secrets = []
    if "://" in url:
        netloc = url.partition("://")[2].partition("/")[0]
        if "@" in netloc:
            userinfo = netloc.rpartition("@")[0]
            _, has_password, password = userinfo.partition(":")
            if has_password and password:
                secrets.append(password)

    _, sep, query = url.partition("?")
    if sep:
        for pair in query.partition("#")[0].split("&"):
            name, has_value, value = pair.partition("=")
            if has_value and value and name.lower() in CREDENTIAL_QUERY_KEYS:
                secrets.append(value)

    return any(secret in text for secret in secrets)
