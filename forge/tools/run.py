"""Process tools: a general shell runner, and a probe pinned to the target binary.

probe_exe is deliberately separate from run: argv[0] is pinned to the
configured executable so the model cannot drift onto some other binary, and
every probe lands in the audit trail the skill verifier reads back.
"""

from __future__ import annotations

from ..sandbox import SandboxError
from .base import Context, Tool, ToolError


class RunCmd(Tool):
    name = "run"
    description = (
        "Run a shell command in the workspace. Hard timeout, truncated output, "
        "dangerous patterns denied. Use it for builds, tests, and fixtures. For "
        "invoking the target binary, prefer probe_exe."
    )
    gated = True
    input_schema = {
        "type": "object",
        "properties": {
            "cmd": {"type": "string"},
            "timeout": {"type": "integer", "description": "Seconds. Defaults to the configured command timeout."},
        },
        "required": ["cmd"],
    }

    def run(self, ctx: Context, cmd: str, timeout: int | None = None) -> str:
        try:
            res = ctx.sandbox.run(str(cmd), timeout=timeout)
        except SandboxError as e:
            raise ToolError(str(e)) from e
        ctx.audit.append({"tool": "run", "cmd": str(cmd)[:300], "exit": res.exit_code})
        return res.render()


class ProbeExe(Tool):
    name = "probe_exe"
    description = (
        "Run the configured target binary with the given arguments. argv[0] is "
        "pinned to the configured executable — supply arguments only, never the "
        "binary name. Use this for every probe of the target so the invocation "
        "lands in the audit trail."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "args": {"type": "string", "description": "Arguments only, e.g. '--help'. May be empty."},
        },
    }

    def run(self, ctx: Context, args: str = "") -> str:
        exe = ctx.cfg.target_exe
        if not exe:
            raise ToolError("no target executable is configured")
        try:
            res = ctx.sandbox.run(f"'{exe}' {args}".strip(), timeout=ctx.cfg.cmd_timeout)
        except SandboxError as e:
            raise ToolError(str(e)) from e
        ctx.audit.append({"tool": "probe_exe", "args": str(args)[:300], "exit": res.exit_code})
        return res.render()
