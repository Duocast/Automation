"""On-disk skill library.

A skill is a folder with a SKILL.md: YAML frontmatter (name, description) plus a
markdown body. Same shape Anthropic uses, so skills written here are portable to
Claude Code and back.

Loading is *progressive disclosure*: only name + description go into the system
prompt. The body is pulled on demand via the skill_read tool. Ten skills cost ~300
tokens of context instead of ~30k, and the model still knows they exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
_NAME_OK = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path
    meta: dict[str, str]

    def render(self) -> str:
        return f"# Skill: {self.name}\n\n{self.body.strip()}\n"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal YAML: flat `key: value` pairs, optional quotes, optional | blocks.

    Deliberately not PyYAML — skills are machine-written here and a 30-line parser
    beats a dependency plus arbitrary-tag deserialisation on model-authored input.
    """
    m = _FM.match(text)
    if not m:
        return {}, text
    raw, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    key: str | None = None
    block: list[str] = []
    for line in raw.splitlines():
        if key and (line.startswith("  ") or not line.strip()):
            block.append(line.strip())
            continue
        if key:
            meta[key] = " ".join(b for b in block if b).strip()
            key, block = None, []
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if v in {"|", ">", "|-", ">-"}:
            key, block = k, []
            continue
        meta[k] = v.strip("'\"")
    if key:
        meta[key] = " ".join(b for b in block if b).strip()
    return meta, body


class SkillStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Skill]:
        out: list[Skill] = []
        for p in sorted(self.root.glob("*/SKILL.md")):
            try:
                meta, body = _parse_frontmatter(p.read_text(encoding="utf-8"))
            except OSError:
                continue
            name = meta.get("name") or p.parent.name
            out.append(Skill(name, meta.get("description", ""), body, p, meta))
        return out

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.list() if s.name == name), None)

    def index(self) -> str:
        """The block injected into the system prompt."""
        skills = self.list()
        if not skills:
            return "(no skills learned yet)"
        lines = []
        for s in skills:
            v = s.meta.get("verified_examples")
            tag = f" [verified: {v}]" if v else " [UNVERIFIED]"
            lines.append(f"- {s.name}{tag}: {s.description}")
        return "\n".join(lines)

    def save(self, name: str, description: str, body: str, meta: dict | None = None) -> Path:
        if not _NAME_OK.match(name):
            raise ValueError(f"bad skill name {name!r}: use lowercase letters, digits, hyphens")
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        fm = {"name": name, "description": description.replace("\n", " ").strip(), **(meta or {})}
        fm.setdefault("updated", date.today().isoformat())
        head = "\n".join(f"{k}: {_yaml_scalar(str(v))}" for k, v in fm.items())
        p = d / "SKILL.md"
        p.write_text(f"---\n{head}\n---\n\n{body.strip()}\n", encoding="utf-8")
        return p


def _yaml_scalar(v: str) -> str:
    if v == "" or re.search(r"[:#\n]|^\s|\s$", v):
        return "'" + v.replace("'", "''") + "'"
    return v
