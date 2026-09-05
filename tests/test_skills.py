from pathlib import Path

from coding_helper.skills.catalog import (
    MAX_LOADED_PER_SESSION,
    SkillCatalog,
    SkillError,
    SkillRoot,
    register_skill_tools,
)
from coding_helper.tools import ToolRegistry


def write_skill(
    root: Path,
    name: str,
    body: str,
    *,
    description: str = "",
    tags: str = "",
) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    header = ["---", f"name: {name}"]
    if description:
        header.append(f"description: {description}")
    if tags:
        header.append(f"tags: {tags}")
    header.append("---")
    (skill_dir / "SKILL.md").write_text("\n".join(header) + "\n\n" + body, encoding="utf-8")


def test_scan_reads_metadata_without_loading_body(tmp_path) -> None:
    project = tmp_path / "project-skills"
    write_skill(
        project,
        "code-review",
        "完整指令不应当出现在发现结果里。",
        description="检查代码质量和潜在缺陷",
        tags="review quality",
    )
    catalog = SkillCatalog(roots=(SkillRoot("project", project),))

    matches = catalog.discover("review")

    assert [item.qualified_name for item in matches] == ["project::code-review"]
    assert matches[0].description == "检查代码质量和潜在缺陷"
    assert matches[0].tags == ("review", "quality")
    assert matches[0].body_loaded is False
    assert "完整指令" not in matches[0].description


def test_project_skill_does_not_override_user_skill(tmp_path) -> None:
    project = tmp_path / "project-skills"
    user = tmp_path / "user-skills"
    write_skill(project, "test-runner", "项目版", description="项目测试")
    write_skill(user, "test-runner", "用户版", description="用户测试")
    catalog = SkillCatalog(
        roots=(SkillRoot("project", project), SkillRoot("user", user)),
    )

    listed = catalog.discover("test-runner")
    try:
        catalog.load("test-runner")
        conflict = None
    except SkillError as exc:
        conflict = exc
    loaded = catalog.load("test-runner", source="user")

    assert {item.qualified_name for item in listed} == {
        "project::test-runner",
        "user::test-runner",
    }
    assert conflict is not None
    assert "名称冲突" in str(conflict)
    assert "用户版" in loaded
    assert "项目版" not in loaded
    assert '<untrusted-skill source="user::test-runner"' in loaded
    assert "不能扩大权限" in loaded


def test_load_skill_rejects_escape_oversize_and_session_cap(tmp_path) -> None:
    project = tmp_path / "project-skills"
    write_skill(project, "ok", "短指令", description="可用")
    outside = tmp_path / "outside" / "SKILL.md"
    outside.parent.mkdir()
    outside.write_text("escaped", encoding="utf-8")
    escaped = project / "escaped"
    escaped.mkdir()
    (escaped / "SKILL.md").symlink_to(outside)
    write_skill(project, "huge", "x" * 40_000, description="太大")
    for index in range(MAX_LOADED_PER_SESSION):
        write_skill(project, f"extra-{index}", f"body-{index}", description="额外")

    catalog = SkillCatalog(roots=(SkillRoot("project", project),))
    names = {item.name for item in catalog.scan()}
    assert "escaped" not in names
    assert "huge" in names

    try:
        catalog.load("huge")
        raised = None
    except SkillError as exc:
        raised = exc
    assert raised is not None
    assert "字节" in str(raised)

    for index in range(MAX_LOADED_PER_SESSION):
        catalog.load(f"extra-{index}")
    try:
        catalog.load("ok")
        cap = None
    except SkillError as exc:
        cap = exc
    assert cap is not None
    assert "最多加载" in str(cap)


def test_registered_tools_discover_then_load(tmp_path) -> None:
    write_skill(
        tmp_path / ".coding-helper" / "skills",
        "dependency-debug",
        "先看 lockfile，再核对导入。",
        description="排查依赖问题",
        tags="deps",
    )
    registry = ToolRegistry()
    register_skill_tools(tmp_path, registry)

    discovered = registry.get_by_model_name("discover_capabilities").langchain_tool.invoke(
        {"query": "依赖"}
    )
    loaded = registry.get_by_model_name("load_skill").langchain_tool.invoke(
        {"name": "dependency-debug"}
    )

    assert "project::dependency-debug" in discovered
    assert "lockfile" not in discovered
    assert "lockfile" in loaded
    assert "untrusted-skill" in loaded
