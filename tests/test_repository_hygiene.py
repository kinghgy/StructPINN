import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_local_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    missing: list[str] = []
    for document in ROOT.rglob("*.md"):
        if any(part in {".pytest_cache", ".ruff_cache", "runs"} for part in document.parts):
            continue
        for target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = target.split("#", 1)[0]
            if path_text and not (document.parent / path_text).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "Missing local Markdown links:\n" + "\n".join(missing)


def test_repository_metadata_parses() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text("utf-8"))
    assert workflow["jobs"]["test"]["steps"]
    assert citation["cff-version"] == "1.2.0"
    for run_configuration in (ROOT / ".run").glob("*.xml"):
        ET.parse(run_configuration)


def test_publishable_tree_contains_no_private_binary_sources() -> None:
    forbidden_suffixes = {".pdf", ".pt", ".pth"}
    unexpected: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {"runs", ".git"} for part in path.parts):
            continue
        if path.suffix.lower() in forbidden_suffixes:
            unexpected.append(str(path.relative_to(ROOT)))
    assert not unexpected, f"Private or binary source files found: {unexpected}"
