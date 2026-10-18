"""Repository hygiene: no tracked file holds a personal home path, and provenance never records one."""
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmarks'))
from data import REPOSITORY, descriptor, provenance, recorded_path

# Every pattern and synthetic value is built from fragments, so this file never matches itself.
ROOTS = ('ho' + 'me', 'Us' + 'ers')
HOME_PATH = re.compile(rb'(?<![A-Za-z0-9._-])/(?:' + b'|'.join(r.encode() for r in ROOTS) + rb')/[A-Za-z0-9._-]+/')
# The one allowed occurrence: the synthetic PATH entry '<home>/user/bin:/usr/bin' of
# test_F11_bad_path_before_tool, matched as an exact token (with the ':'), once in the whole tree.
EXC_FILE = 'tests/unit/test_conversion_fix_r002.py'
EXC_TOKEN = b'/' + b'ho' + b'me' + b'/user/bin' + b':'
LOCAL_POINTERS = ('CLAUDE.local.md', '.agents/rules/90-local.md')


def _allowed(name, data, start):
    return name == EXC_FILE and data.startswith(EXC_TOKEN, start)


def home_path_hits(name, data):
    """Offsets of home paths in data, except the one allowed token."""
    return [m.start() for m in HOME_PATH.finditer(data) if not _allowed(name, data, m.start())]


def exception_matches(name, data):
    return sum(_allowed(name, data, m.start()) for m in HOME_PATH.finditer(data))


def home_path_failures(items):
    """items: (repository-relative name, bytes) pairs; returns the failures as text."""
    failures, allowed = [], 0
    for name, data in items:
        hits = home_path_hits(name, data)
        if hits:
            failures.append(f'{name}:{len(hits)}')
        allowed += exception_matches(name, data)
    if allowed == 0:
        failures.append('stale exception')
    elif allowed > 1:
        failures.append(f'exception matched {allowed} times')
    return failures


def tracked_files():
    # No skip: without git the guard cannot run, and that is a failure.
    out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], check=True, capture_output=True).stdout
    return [name for name in out.decode('utf-8').split('\0') if name]


def _home(rest):
    return b'/' + ROOTS[0].encode() + rest


def _r002_bytes():
    return (ROOT / EXC_FILE).read_bytes()


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / 'home-dir'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))
    return home


def test_repository_is_the_clone_root():
    assert REPOSITORY == ROOT
    assert (REPOSITORY / 'pyproject.toml').is_file()


def test_recorded_path_inside_repository_is_relative():
    assert recorded_path(REPOSITORY / 'scripts/benchmarks/data.py') == 'scripts/benchmarks/data.py'
    assert recorded_path(str(REPOSITORY / 'tests/fixtures')) == 'tests/fixtures'


def test_recorded_path_repository_root_is_dot():
    assert recorded_path(REPOSITORY) == '.'


def test_recorded_path_outside_home_and_repository_is_unchanged(tmp_path, fake_home):
    system = tmp_path / 'system' / 'fonts-notice.txt'
    assert recorded_path(system) == str(system.resolve())


def test_recorded_path_under_home_outside_repository_raises(fake_home):
    with pytest.raises(ValueError):
        recorded_path(fake_home / 'someone' / 'annotations.json')


def test_recorded_path_symlink_alias_of_repository(tmp_path, monkeypatch):
    # The alias lies under the (fake) home: the path and REPOSITORY resolve the same way.
    monkeypatch.setattr(Path, 'home', staticmethod(lambda: tmp_path))
    link = tmp_path / 'clone'
    link.symlink_to(REPOSITORY, target_is_directory=True)
    assert recorded_path(link / 'scripts/benchmarks/data.py') == 'scripts/benchmarks/data.py'


def test_recorded_path_symlink_leaving_repository_records_target(tmp_path, fake_home):
    outside = tmp_path / 'outside'
    outside.mkdir()
    link = tmp_path / 'to-outside'
    link.symlink_to(outside, target_is_directory=True)
    assert recorded_path(link / 'x.json') == str(outside.resolve() / 'x.json')
    to_home = tmp_path / 'to-home'
    to_home.symlink_to(fake_home, target_is_directory=True)
    with pytest.raises(ValueError):
        recorded_path(to_home / 'x.json')


def test_provenance_keeps_identity_and_records_relative_path():
    path = REPOSITORY / 'LICENSE'
    assert provenance(path) == dict(descriptor(path), path='LICENSE')
    assert provenance(str(path))['sha256'] == descriptor(path)['sha256']


def test_provenance_refuses_home_file(fake_home):
    personal = fake_home / 'notes.json'
    personal.write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError):
        provenance(personal)


def test_scanner_detects_home_and_users_paths():
    assert len(home_path_hits('notes.md', b'x ' + _home(b'/someone/y'))) == 1
    assert len(home_path_hits('notes.md', b'x /' + ROOTS[1].encode() + b'/someone/y')) == 1


def test_scanner_allows_only_the_r002_token():
    data = _r002_bytes()
    assert home_path_hits(EXC_FILE, data) == []
    assert exception_matches(EXC_FILE, data) == 1
    assert len(home_path_hits('tests/unit/other.py', data)) == 1
    assert len(home_path_hits(EXC_FILE, data + b"\n'" + _home(b'/user2/x') + b"'\n")) == 1


def test_scanner_ignores_word_list_in_a003_report():
    data = (ROOT / 'documentation/benchmarks/a003-engine-comparison-r003.md').read_bytes()
    assert home_path_hits('documentation/benchmarks/a003-engine-comparison-r003.md', data) == []


def test_exception_boundary_and_count():
    data = _r002_bytes()
    assert data.count(EXC_TOKEN) == 1
    binary = data.replace(EXC_TOKEN, _home(b'/user/binary:'))
    assert len(home_path_hits(EXC_FILE, binary)) == 1 and exception_matches(EXC_FILE, binary) == 0
    subdir = data.replace(EXC_TOKEN, _home(b'/user/bin/x:'))
    assert len(home_path_hits(EXC_FILE, subdir)) == 1
    assert home_path_failures([(EXC_FILE, data.replace(EXC_TOKEN, b''))]) == ['stale exception']
    twice = data.replace(EXC_TOKEN, EXC_TOKEN + b' ' + EXC_TOKEN)
    assert home_path_failures([(EXC_FILE, twice)]) == ['exception matched 2 times']


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
