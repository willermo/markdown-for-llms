"""Helpers of the bench tests. Bench tests never skip (`scripts/bench_junit_guard.py`).

Tests that use the external tools run them for real in the reference toolchain image, with the
runner of the bench (`--network none`, the binaries of the measures). `require_toolchain` makes
such a test fail, saying how to build the image, when it is missing or differs from
`toolchain.json`. `FakeEngine` stands in for the Docker client only for the branches a real tool
cannot be made to take on demand (exit 137, 125-127, timeouts and their cleanup).
"""
import json
import subprocess

from markdown_for_llms.bench import toolchain
from markdown_for_llms.bench.codes import BenchError
from markdown_for_llms.bench.selftest import growing_html, make_pdf  # noqa: F401  (used by the tests)

FAKE_IMAGE = "sha256:" + "f" * 64


def require_toolchain():
    """ID of the verified reference image, or a failure (never a skip) that says how to build it."""
    try:
        return toolchain.image()
    except BenchError as exc:
        raise AssertionError(f"reference toolchain image: {exc.code.value} ({exc.detail}); build it with "
                             "`mdf bench prepare --toolchain` at the root of the checkout") from None


class FakeEngine:
    """A Docker client that records its arguments; `run` answers with `result(argv, stdin)`, which
    returns (exit status, stdout, stderr) or raises (for example `subprocess.TimeoutExpired`)."""

    def __init__(self, result=None, remove_ok=True, image=None):
        self.calls, self.result, self.remove_ok, self.image = [], result, remove_ok, image

    def __call__(self, args, timeout, stdin=b""):
        self.calls.append(list(args))
        verb = args[0]
        if verb == "run":
            if args[-2:] == ["cat", toolchain.MANIFEST_PATH]:
                return self._done(0, self.image.get("manifest", b"") if self.image else b"")
            rc, out, err = self.result(args, stdin) if self.result else (0, b"", b"")
            return self._done(rc, out, err)
        if verb == "rm":
            return self._done(0)
        if verb == "container":
            return self._done(1, b"", b"Error response from daemon: No such container: x") if self.remove_ok \
                else self._done(0, b"[{}]")
        if verb == "image":
            if not self.image:
                return self._done(1, b"", b"Error: No such image")
            return self._done(0, json.dumps({"Id": self.image["id"], "Config": self.image["config"]}).encode())
        if verb == "version":
            return self._done(0, b"0.0.test\n")
        if verb == "context":
            return self._done(0, b"unix:///var/run/docker.sock\n")
        if verb == "ps":
            return self._done(0, b"")
        raise AssertionError("unexpected docker call: " + " ".join(args))

    @staticmethod
    def _done(rc, out=b"", err=b""):
        return subprocess.CompletedProcess([], rc, out, err)

    def runs(self):
        return [c for c in self.calls if c[0] == "run"]
