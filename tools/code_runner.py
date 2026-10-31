# """Runs a short Python snippet in a separate process with a time limit.

# This is a *light* sandbox: separate process, empty temp folder, no
# environment variables, isolated mode (-I), 5 second limit, output capped.
# It is NOT safe against a determined attacker. That's why tools are off by
# default. On Linux (e.g. a Hugging Face Space) we also cap memory and CPU.
# """

# import os
# import subprocess
# import sys
# import tempfile

# MAX_OUTPUT = 2000


# def _limit_resources():  # only runs on Linux/macOS
#     import resource

#     resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
#     resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))


# def run_python(code: str, timeout: int = 5) -> str:
#     if not code.strip():
#         return "No code given."
#     with tempfile.TemporaryDirectory() as tmp:
#         path = os.path.join(tmp, "snippet.py")
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(code)
#         kwargs = {}
#         if os.name == "posix":
#             kwargs["preexec_fn"] = _limit_resources
#         try:
#             proc = subprocess.run(
#                 [sys.executable, "-I", path],
#                 cwd=tmp,
#                 capture_output=True,
#                 text=True,
#                 timeout=timeout,
#                 # SYSTEMROOT is needed for Python to start on Windows
#                 env={k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT")},
#                 **kwargs,
#             )
#         except subprocess.TimeoutExpired:
#             return f"Timed out after {timeout} s (possible infinite loop)."
#     out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
#     out = out.strip() or "(no output)"
#     if len(out) > MAX_OUTPUT:
#         out = out[:MAX_OUTPUT] + "\n...(output cut)"
#     return f"exit code {proc.returncode}\n{out}"
