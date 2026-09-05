from pypdf import PdfWriter

from coding_helper.context.references import (
    MAX_PINNED_CHARS,
    attach_pinned_context,
    parse_at_references,
)


MINIMAL_TEXT_PDF = b"""%PDF-1.1
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144]
   /Contents 4 0 R
   /Resources << /Font << /F1 5 0 R >> >>
>>
endobj
4 0 obj
<< /Length 55 >>
stream
BT /F1 12 Tf 20 100 Td (login refresh token) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
0000000380 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
459
%%EOF
"""


def test_parse_at_references_skips_email_and_supports_quotes() -> None:
    text = '联系 user@example.com，并阅读 @"my notes.txt" 与 @src/auth.py.'
    parsed = parse_at_references(text)

    assert [item.raw for item in parsed] == ["my notes.txt", "src/auth.py"]


def test_attach_pinned_context_injects_text_directory_and_pdf(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    notes = workspace / "docs"
    notes.mkdir(parents=True)
    (notes / "my notes.txt").write_text("alpha=1\n", encoding="utf-8")
    (workspace / "src").mkdir()
    (workspace / "src" / "auth.py").write_text("TOKEN = 'old'\n", encoding="utf-8")
    (workspace / "spec.pdf").write_bytes(MINIMAL_TEXT_PDF)

    pinned = attach_pinned_context(
        '分析 @"docs/my notes.txt"、@src/ 和 @spec.pdf',
        workspace,
    )

    assert len(pinned.artifacts) == 3
    assert pinned.artifacts[0].media_type == "text/plain"
    assert "1: alpha=1" in pinned.artifacts[0].content
    assert pinned.artifacts[1].media_type == "text/directory-manifest"
    assert "auth.py" in pinned.artifacts[1].content
    assert pinned.artifacts[2].media_type == "application/pdf"
    assert "--- page 1 ---" in pinned.artifacts[2].content
    assert "login refresh token" in pinned.artifacts[2].content
    assert "<pinned-context>" in pinned.prompt
    assert "不可信数据" in pinned.prompt


def test_attach_pinned_context_rejects_escape_and_sensitive_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")

    pinned = attach_pinned_context("不要读 @../outside.txt 或 @.env", workspace)

    assert [item.error is not None for item in pinned.artifacts] == [True, True]
    assert "secret" not in pinned.prompt
    assert "TOKEN=secret" not in pinned.prompt
    assert "超出 Workspace" in pinned.prompt or "路径不存在" in pinned.prompt
    assert "敏感文件" in pinned.prompt


def test_large_text_file_is_truncated_and_directory_omits_cache(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    src = workspace / "pkg"
    src.mkdir(parents=True)
    (src / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / ".venv").mkdir()
    (workspace / ".venv" / "hidden.py").write_text("nope", encoding="utf-8")
    huge = workspace / "big.txt"
    huge.write_text(("x" * 80 + "\n") * 200, encoding="utf-8")

    pinned = attach_pinned_context("看 @big.txt 和 @pkg/", workspace)
    text, directory = pinned.artifacts

    assert text.truncated is True
    assert len(text.content) <= MAX_PINNED_CHARS
    assert "app.py" in directory.content
    assert ".venv" not in directory.content


def test_pdf_errors_for_encrypted_and_textless_files(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    blank = PdfWriter()
    blank.add_blank_page(width=72, height=72)
    blank_path = workspace / "blank.pdf"
    blank.write(blank_path)

    locked = PdfWriter()
    locked.add_blank_page(width=72, height=72)
    locked.encrypt("not-a-real-secret")
    locked_path = workspace / "locked.pdf"
    locked.write(locked_path)

    pinned = attach_pinned_context("总结 @blank.pdf 和 @locked.pdf", workspace)

    assert pinned.artifacts[0].error is not None
    assert "OCR" in pinned.artifacts[0].error
    assert pinned.artifacts[1].error is not None
    assert "加密" in pinned.artifacts[1].error
    assert "not-a-real-secret" not in pinned.prompt


def test_unknown_mention_is_ignored_when_it_is_not_a_path(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    pinned = attach_pinned_context("请 @reviewer 先看一眼", workspace)

    assert pinned.artifacts == ()
    assert pinned.prompt == "请 @reviewer 先看一眼"
