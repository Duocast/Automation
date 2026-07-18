"""Source acquisition. Offline-first: a local checkout is always accepted;
a git URL only when the run is not in offline mode."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..sandbox import SandboxError
from .base import Context, Tool, ToolError, rel

_IGNORE = shutil.ignore_patterns(
    ".git", "node_modules", "__pycache__", ".venv", "venv", "target", "dist", ".mypy_cache"
)


class FetchRepo(Tool):
    name = "fetch_repo"
    description = (
        "Fetch the project's source tree into the workspace (default ./repo) so you "
        "can read and grep it. Accepts a local directory path always; a git URL only "
        "when online. Defaults to the configured target repo."
    )
    gated = True
    input_schema = {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Local path or git URL. Defaults to the configured repo."},
            "dest": {"type": "string", "description": "Destination inside the workspace.", "default": "repo"},
        },
    }

    def run(self, ctx: Context, repo: str | None = None, dest: str = "repo") -> str:
        src = str(repo or ctx.cfg.target_repo or "").strip()
        if not src:
            raise ToolError("no repo given and none configured")
        try:
            d = ctx.sandbox.resolve_write(dest)
        except SandboxError as e:
            raise ToolError(str(e)) from e

        local = Path(src).expanduser()
        if local.is_dir():
            shutil.copytree(local, d, ignore=_IGNORE, dirs_exist_ok=True)
            ctx.audit.append({"tool": "fetch_repo", "src": str(local), "mode": "copy"})
            return f"copied local checkout {local} -> {rel(ctx, d)}/. Explore it with list_dir."

        if ctx.cfg.offline:
            raise ToolError(
                f"offline mode: {src!r} is not a local directory. Clone the repo yourself "
                "and point fetch_repo (or --repo) at the local checkout."
            )
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", src, str(d)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise ToolError(f"git clone failed: {proc.stderr.strip()[-800:]}")
        ctx.audit.append({"tool": "fetch_repo", "src": src, "mode": "clone"})
        return f"cloned {src} -> {rel(ctx, d)}/. Explore it with list_dir."
