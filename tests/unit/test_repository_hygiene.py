"""Repository hygiene: no tracked file holds a personal home path, and local pointer files stay untracked."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]

# Every pattern and synthetic value is built from fragments, so this file never matches itself.
ROOTS = ('ho' + 'me', 'Us' + 'ers')
HOME_PATH = re.compile(rb'(?<![A-Za-z0-9._-])/(?:' + b'|'.join(r.encode() for r in ROOTS) + rb')/[A-Za-z0-9._-]+/')
LOCAL_POINTERS = ('CLAUDE.local.md', '.agents/rules/90-local.md')


def home_path_hits(name, data):
    """Offsets of home paths in data."""
    return [m.start() for m in HOME_PATH.finditer(data)]


def home_path_failures(items):
    """items: (repository-relative name, bytes) pairs; returns the failures as text."""
    failures = []
    for name, data in items:
        hits = home_path_hits(name, data)
        if hits:
            failures.append(f'{name}:{len(hits)}')
    return failures


def tracked_files():
    # No skip: without git the guard cannot run, and that is a failure.
    out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], check=True, capture_output=True).stdout
    return [name for name in out.decode('utf-8').split('\0') if name]


def _home(rest):
    return b'/' + ROOTS[0].encode() + rest


def test_scanner_detects_home_and_users_paths():
    assert len(home_path_hits('notes.md', b'x ' + _home(b'/someone/y'))) == 1
    assert len(home_path_hits('notes.md', b'x /' + ROOTS[1].encode() + b'/someone/y')) == 1
    assert home_path_failures([('notes.md', b'x ' + _home(b'/someone/y') + b' ' + _home(b'/other/z'))]) == ['notes.md:2']


def test_scanner_ignores_word_list_fragment():
    # A slash-separated word list such as 'clone/gold/<home>/temp/antenati' is not a home path.
    data = b'| CLA-I005 | Allowlist; clone/gold/' + ROOTS[0].encode() + b'/temp/antenati scratch rejected. |\n'
    assert home_path_hits('notes.md', data) == []


def test_tracked_files_have_no_home_paths():
    items = []
    for name in tracked_files():
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            items.append((name, path.read_bytes()))
    assert items
    assert home_path_failures(items) == []


def test_local_pointer_files_are_not_tracked():
    tracked = set(tracked_files())
    assert not tracked & set(LOCAL_POINTERS)
