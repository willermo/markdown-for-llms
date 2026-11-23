"""Keys from an env file: `DATALAB_API_KEY`, the keys of the inference servers (MFL033).

`read(env_file, names)` opens the file without following a link and without blocking (a FIFO
given by mistake is refused, not waited on), and accepts only a regular file of the caller with
mode 0600 and at most `LIMIT` bytes; otherwise `Unsafe`. It returns {name: value} for the names
it finds (`NAME=value`, `export NAME=value`, quotes stripped, `#` comments and blank lines
skipped). A value never appears in a repr, a log or an error: `Secret` wraps it.
Standard library only.
"""
import os
import stat

LIMIT = 65536
# The suffixes of the names of keys (MFL046 c): the names of `key_env` of servers.json end with one, so that
# `mdf bench secret-scan`, which searches every variable of the env file with one of them, finds every key.
KEY_SUFFIXES = ("_KEY", "_SECRET", "_TOKEN", "_TOKEN_ID", "_BEARER")


class Unsafe(Exception):
    """The env file is not a private regular file of the caller within the limit."""


class Secret:
    """A key; its value never appears in a repr, a log or an error."""
    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return "Secret(***)"

    __str__ = __repr__

    def reveal(self):
        return self._value


def parse(data, names):
    """{name: value} of the lines of `data` (text) that set one of `names`; the first wins."""
    found = {}
    for line in data.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if sep and name in names and name not in found:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            found[name] = value
    return found


def read(env_file, names):
    """{name: value} from `env_file` for the `names` it sets with a non-empty value (every name
    when `names` is None)."""
    try:
        fd = os.open(env_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as exc:
        raise Unsafe("env file: %s" % type(exc).__name__) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise Unsafe("env file: a regular file of the caller with mode 0600")
        data = os.read(fd, LIMIT + 1)
    finally:
        os.close(fd)
    if len(data) > LIMIT:
        raise Unsafe("env file too large")
    text = data.decode("utf-8", "replace")
    if names is None:
        names = {line.strip().removeprefix("export ").partition("=")[0].strip() for line in text.splitlines()}
    return {k: v for k, v in parse(text, set(names)).items() if v}
