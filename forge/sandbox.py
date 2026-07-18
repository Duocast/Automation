"""Containment for filesystem and process access.

The agent writes files and runs a binary. Two rules make that tolerable:
  1. Every path is resolved and must land inside an allowed root. Resolution happens
     BEFORE the check, so `../../etc/passwd` and symlink escapes both fail.
  2. Every command is matched against a denylist and runs under a hard timeout with
     truncated output.

This is defence in depth, not a jail. It stops an agent that has confused itself.
It does not stop a determined attacker who controls the prompt. For untrusted input,
run the whole process in a container with no credentials mounted.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SandboxError(Exception):
    """Raised when the agent tries to leave the box. Returned to the model as a
    tool error so it can correct course, not crashed."""


# Patterns that are never worth the risk. Ordered roughly by how bad the day gets.
DENY_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+/(?:\s|$)", "recursive delete of /"),
    (r"\brm\s+-[a-zA-Z]*[rf].*\s(/|~|\$HOME)(\s|/?$)", "recursive delete of a home/root path"),
    (r"\bmkfs(\.\w+)?\b", "filesystem format"),
    (r"\bdd\b.*\bof=/dev/", "raw device write"),
    (r">\s*/dev/(sd|nvme|disk)", "raw device write"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "host power control"),
    (r"\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?(ba)?sh", "pipe-to-shell"),
    (r"\bchmod\s+-R\s+777\s+/", "recursive world-writable on /"),
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:", "fork bomb"),
    (r"\bhistory\s+-c\b", "history wipe"),
    (r"\bgit\b.*\bpush\b.*--force", "force push"),
    (r"\bsudo\b", "privilege escalation"),
]

# Commands that reach the network. Blocked unless explicitly allowed, because a
# learned skill should be derived from the repo and the binary, not from a live
# endpoint that can change under us.
NETWORK_BINS = {"curl", "wget", "nc", "ncat", "telnet", "ssh", "scp", "rsync", "ftp"}

_ENV_DENY = {"ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "OPENAI_API_KEY"}


@dataclass
class CmdResult:
    cmd: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False

    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def render(self, limit: int = 30_000) -> str:
        head = f"$ {self.cmd}\nexit={self.exit_code}"
        if self.timed_out:
            head += " (TIMED OUT)"
        body = ""
        if self.stdout:
            body += f"\n--- stdout ---\n{self.stdout}"
        if self.stderr:
            body += f"\n--- stderr ---\n{self.stderr}"
        if not body:
            body = "\n(no output)"
        out = head + body
        if len(out) > limit:
            out = out[:limit] + f"\n... [truncated at {limit} chars]"
        return out


class Sandbox:
    def __init__(
        self,
        workspace: Path,
        read_roots: list[Path] | None = None,
        allow_network: bool = False,
        default_timeout: int = 60,
        write_allowlist: list[Path] | None = None,
    ):
        self.workspace = Path(workspace).resolve()
        self.read_roots = [Path(p).resolve() for p in (read_roots or [])]
        self.allow_network = allow_network
        self.default_timeout = default_timeout
        # None means "anywhere in the workspace". A list confines writes to those
        # paths (a directory covers its subtree) — how a subagent's declared
        # write set is enforced rather than trusted. Never loosens the workspace
        # boundary: an allowlist entry outside the workspace grants nothing.
        self.write_allowlist = (
            None if write_allowlist is None
            else [Path(p).resolve() for p in write_allowlist]
        )
        self.workspace.mkdir(parents=True, exist_ok=True)

    # --- paths -------------------------------------------------------------
    def _resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        # strict=False so we can resolve paths for files we're about to create
        return p.resolve()

    def resolve_write(self, path: str | Path) -> Path:
        p = self._resolve(path)
        if not self._within(p, self.workspace):
            raise SandboxError(
                f"write denied: {p} is outside the workspace ({self.workspace}). "
                "Write inside the workspace."
            )
        if self.write_allowlist is not None and not any(
            self._within(p, a) for a in self.write_allowlist
        ):
            allowed = ", ".join(str(a) for a in self.write_allowlist)
            raise SandboxError(
                f"write denied: {p} is outside your declared write set ({allowed}). "
                "Write only the paths you declared, or use your scratch directory."
            )
        return p

    def resolve_read(self, path: str | Path) -> Path:
        p = self._resolve(path)
        roots = [self.workspace, *self.read_roots]
        if not any(self._within(p, r) for r in roots):
            allowed = ", ".join(str(r) for r in roots)
            raise SandboxError(f"read denied: {p} is outside the allowed roots ({allowed}).")
        return p

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    # --- commands ----------------------------------------------------------
    def check_command(self, cmd: str) -> None:
        for pattern, why in DENY_PATTERNS:
            if re.search(pattern, cmd):
                raise SandboxError(f"command denied ({why}): {cmd!r}")
        if not self.allow_network:
            for token in self._binaries(cmd):
                if token in NETWORK_BINS:
                    raise SandboxError(
                        f"command denied (network access via {token!r} is disabled). "
                        "Use the fetch_repo tool or web_search instead."
                    )

    @staticmethod
    def _binaries(cmd: str) -> set[str]:
        """Best-effort: first token of each pipeline/list segment."""
        out: set[str] = set()
        for seg in re.split(r"[|;&]+|\$\(|`", cmd):
            seg = seg.strip()
            if not seg:
                continue
            try:
                parts = shlex.split(seg)
            except ValueError:
                parts = seg.split()
            for part in parts:
                if part in {"sudo", "env", "nohup", "time", "xargs"}:
                    continue
                out.add(Path(part).name)
                break
        return out

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _ENV_DENY}
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("NO_COLOR", "1")   # keep ANSI noise out of the context window
        env.setdefault("TERM", "dumb")
        return env

    def run(
        self,
        cmd: str,
        timeout: int | None = None,
        cwd: Path | None = None,
        max_chars: int = 30_000,
    ) -> CmdResult:
        self.check_command(cmd)
        wd = self.resolve_write(cwd) if cwd else self.workspace
        wd.mkdir(parents=True, exist_ok=True)
        t = timeout or self.default_timeout
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(wd),
                env=self._env(),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=t,
                # Never let a subprocess block forever waiting on stdin.
                stdin=subprocess.DEVNULL,
            )
            out, err, code, to = proc.stdout, proc.stderr, proc.returncode, False
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            code, to = 124, True
            err += f"\n[killed after {t}s]"

        truncated = False
        half = max_chars // 2
        if len(out) > max_chars:
            out = out[:half] + f"\n... [{len(out) - max_chars} chars elided] ...\n" + out[-half:]
            truncated = True
        if len(err) > max_chars:
            err = err[:half] + f"\n... [{len(err) - max_chars} chars elided] ...\n" + err[-half:]
            truncated = True
        return CmdResult(cmd, code, out.strip(), err.strip(), to, truncated)
