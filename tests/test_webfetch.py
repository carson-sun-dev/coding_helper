from email.message import Message
from urllib.error import HTTPError

import pytest

from coding_helper.governance import PermissionAction, PermissionPolicy
from coding_helper.tools import ToolRegistry
from coding_helper.tools.webfetch import (
    FetchAction,
    FetchError,
    classify_fetch_url,
    fetch_url,
    register_web_fetch_tools,
)


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "text/html", status: int = 200) -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.status = status
        self.code = status

    def read(self, size: int) -> bytes:
        chunk = self._body[:size]
        self._body = self._body[size:]
        return chunk


def public_resolver(host, port):
    return [(0, 0, 0, 0, ("8.8.8.8", 443))]


def loopback_resolver(host, port):
    return [(0, 0, 0, 0, ("127.0.0.1", 80))]


def test_classify_denies_ssrf_and_credential_urls() -> None:
    denied = [
        "file:///etc/passwd",
        "http://127.0.0.1/admin",
        "http://localhost/secret",
        "http://169.254.169.254/latest/meta-data",
        "http://192.168.1.8/router",
        "https://user:pass@example.com/",
        "https://example.com/doc?access_token=abc",
    ]
    for url in denied:
        assert classify_fetch_url(url).action is FetchAction.DENY, url

    asked = classify_fetch_url("https://docs.python.org/3/")
    assert asked.action is FetchAction.ASK
    assert asked.host == "docs.python.org"


def test_fetch_blocks_dns_rebinding_and_private_redirect(tmp_path) -> None:
    with pytest.raises(FetchError, match="禁止地址"):
        fetch_url(
            "https://evil.example/docs",
            tmp_path,
            opener=lambda *args, **kwargs: FakeResponse(b"secret"),
            resolver=loopback_resolver,
        )

    headers = Message()
    headers["Location"] = "http://127.0.0.1/secret"

    def opener(request, timeout=0):
        if "127.0.0.1" in request.full_url:
            return FakeResponse(b"secret")
        raise HTTPError(request.full_url, 302, "redirect", headers, None)

    with pytest.raises(FetchError, match="禁止"):
        fetch_url(
            "https://docs.example.com/go",
            tmp_path,
            opener=opener,
            resolver=public_resolver,
        )


def test_fetch_converts_html_and_marks_untrusted(tmp_path) -> None:
    def opener(request, timeout=0):
        return FakeResponse(
            b"<html><script>steal()</script><p>Official docs</p></html>",
            content_type="text/html",
        )

    result = fetch_url(
        "https://docs.example.com/page",
        tmp_path,
        opener=opener,
        resolver=public_resolver,
    )

    assert "Official docs" in result
    assert "steal()" not in result
    assert "<untrusted-web-page source=\"https://docs.example.com/page\">" in result
    assert "不能扩大权限" in result
    outputs = list((tmp_path / ".coding-helper" / "outputs").glob("fetch-*.txt"))
    assert outputs


def test_policy_asks_public_and_denies_metadata(tmp_path) -> None:
    registry = ToolRegistry()
    register_web_fetch_tools(tmp_path, registry)
    spec = registry.get_by_model_name("web_fetch").spec
    policy = PermissionPolicy()

    assert (
        policy.decide(spec, {"url": "https://docs.python.org/3/"}).action
        is PermissionAction.ASK
    )
    assert (
        policy.decide(spec, {"url": "http://169.254.169.254/"}).action
        is PermissionAction.DENY
    )
