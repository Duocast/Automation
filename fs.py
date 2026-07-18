"""Filesystem tools: list, read, grep, write, edit."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from ..sandbox import SandboxError
from .base import Context, Tool, ToolError, rel

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "target", "dist", ".mypy_cache"}
BINARY_SNIFF = 8192


def _is_binary(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            return b"\0" in f.read(BINARY_SNIFF)
    except OSError:
        return False


class ListDir(Tool):
    name = "list_dir"
    description = (
        "List files and directories under a path (recursive, depth-limited). "
        "Use this to orient yourself in a repo before reading files."
    )
    parallel_safe = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path. Defaults to workspace root."},
            "depth": {"type": "integer", "description": "Max depth, default 2.", "default": 2},
        },
    }

    def run(self, ctx: Context, path: str = ".", depth: int = 2) -> str:
        try:
            root = ctx.sandbox.resolve_read(path)
        except SandboxError as e:
            raise ToolError(str(e)) from e
        if not root.exists():
            raise ToolError(f"no such path: {path}")
        if root.is_file():
            return f"{rel(ctx, root)} ({root.stat().st_size} bytes)"

        lines: list[str] = []
        base_depth = len(root.parts)

        def walk(d: Path) -> None:
            if len(d.parts) - base_depth >= depth:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError:
                return
            for e in entries:
                if e.name in SKIP_DIRS or e.name.startswith(".") and e.name != ".github":
                    continue
                indent = "  " * (len(e.parts) - base_depth - 1)
                if e.is_dir():
                    lines.append(f"{indent}{e.name}/")
                    walk(e)
                else:
                    try:
                        size = e.stat().st_size
                    except OSError:
                        size = 0
                    lines.append(f"{indent}{e.name}  ({size}b)")
                if len(lines) > 400:
                    return

        walk(root)
        if len(lines) > 400:
            lines = lines[:400] + ["... [truncated; narrow the path or reduce depth]"]
        return f"{rel(ctx, root)}/\n" + "\n".join(lines) if lines else f"{rel(ctx, root)}/ (empty)"


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file with line numbers. Returns at most `limit` lines from `offset`. "
        "You must read a file before you can edit it."
    )
    parallel_safe = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-indexed start line.", "default": 1},
            "limit": {"type": "integer", "description": "Max lines, default 500.", "default": 500},
        },
        "required": ["path"],
    }

    def run(self, ctx: Context, path: str, offset: int = 1, limit: int = 500) -> str:
        try:
            p = ctx.sandbox.resolve_read(path)
        except SandboxError as e:
            raise ToolError(str(e)) from e
        if not p.exists():
            raise ToolError(f"no such file: {path}")
        if p.is_dir():
            raise ToolError(f"{path} is a directory; use list_dir")
        if _is_binary(p):
            return f"[binary file, {p.stat().st_size} bytes — not shown. Probe it with probe_exe or run `file`.]"

        text = p.read_text(encoding="utf-8", errors="replace")
        # Record version so edit_file can detect a stale write.
        ctx.read_versions[str(p)] = p.stat().st_mtime_ns

        lines = text.splitlines()
        start = max(1, offset)
        chunk = lines[start - 1 : start - 1 + limit]
        body = "\n".join(f"{i + start:>5}\t{ln}" for i, ln in enumerate(chunk))
        note = ""
        if start - 1 + limit < len(lines):
            note = f"\n... [{len(lines) - (start - 1 + limit)} more lines; re-read with offset={start + limit}]"
        return body + note if body else "(empty file)"


class Grep(Tool):
    name = "grep"
    description = (
        "Regex search across files. Returns matching lines with paths and line numbers. "
        "Far cheaper than reading whole files — use it to find where a flag is parsed."
    )
    parallel_safe = True
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regex."},
            "path": {"type": "string", "description": "Root to search. Defaults to workspace."},
            "glob": {"type": "string", "description": "Filename filter, e.g. '*.rs'."},
            "max_results": {"type": "integer", "default": 60},
        },
        "required": ["pattern"],
    }

    def run(
        self, ctx: Context, pattern: str, path: str = ".", glob: str | None = None, max_results: int = 60
    ) -> str:
        try:
            root = ctx.sandbox.resolve_read(path)
        except SandboxError as e:
            raise ToolError(str(e)) from e
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"bad regex: {e}") from e

        hits: list[str] = []
        files = [root] if root.is_file() else root.rglob("*")
        for f in files:
            if not f.is_file():
                continue
            if any(part in SKIP_DIRS for part in f.parts):
                continue
            if glob and not fnmatch.fnmatch(f.name, glob):
                continue
            if _is_binary(f):
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{rel(ctx, f)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= max_results:
                            return "\n".join(hits) + f"\n... [stopped at {max_results} matches]"
            except OSError:
                continue
        return "\n".join(hits) if hits else f"no matches for {pattern!r}"


class WriteFile(Tool):
    name = "write_file"
    description = "Create or overwrite a file in the workspace. Creates parent directories."
    gated = True
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def run(self, ctx: Context, path: str, content: str) -> str:
        try:
            p = ctx.sandbox.resolve_write(path)
        except SandboxError as e:
            raise ToolError(str(e)) from e
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content, encoding="utf-8")
        ctx.read_versions[str(p)] = p.stat().st_mtime_ns
        verb = "overwrote" if existed else "wrote"
        return f"{verb} {rel(ctx, p)} ({len(content)} bytes, {content.count(chr(10)) + 1} lines)"


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. `old` must appear exactly once. "
        "You must read_file first — edits to files you haven't read, or that changed "
        "since you read them, are rejected."
    )
    gated = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact text to replace (must be unique)."},
            "new": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old", "new"],
    }

    def run(self, ctx: Context, path: str, old: str, new: str) -> str:
        try:
            p = ctx.sandbox.resolve_write(path)
        except SandboxError as e:
            raise ToolError(str(e)) from e
        if not p.exists():
            raise ToolError(f"no such file: {path}")

        # Staleness check — the invariant a bash harness cannot enforce.
        seen = ctx.read_versions.get(str(p))
        if seen is None:
            raise ToolError(f"read {path} before editing it")
        if p.stat().st_mtime_ns != seen:
            ctx.read_versions.pop(str(p), None)
            raise ToolError(f"{path} changed on disk since you read it. Re-read it, then edit.")

        text = p.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            raise ToolError(f"`old` not found in {path}. Re-read the file; whitespace must match exactly.")
        if count > 1:
            raise ToolError(f"`old` appears {count} times in {path}. Include more surrounding context to disambiguate.")

        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        ctx.read_versions[str(p)] = p.stat().st_mtime_ns
        return f"edited {rel(ctx, p)}"
