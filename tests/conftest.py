#!/usr/bin/env python3
"""Pytest configuration: network guard and installation preflight before the collection."""

import pytest
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Guardie installate prima della collection; socketpair AF_UNIX resta locale.
_guard_originals = {}

def _deny_network(*args, **kwargs):
    raise RuntimeError('Rete e socket daemon vietati nei test; preparare cache fuori suite')


def pytest_addoption(parser):
    parser.addoption('--test-mode', choices=('official', 'standalone'),
                     default=None, help='Modalità esplicita S/I: default official, oppure standalone locale')


def pytest_configure(config):
    import socket
    import subprocess
    for name in ('connect', 'connect_ex', 'sendto'):
        _guard_originals[name] = getattr(socket.socket, name)
        setattr(socket.socket, name, _deny_network)
    source = os.environ.get('RUN_SOURCE_MANIFEST')
    receipt = os.environ.get('RUN_INSTALLATION_RECEIPT')
    if not source or not receipt:
        raise pytest.UsageError('Preparazione mancante: RUN_SOURCE_MANIFEST e RUN_INSTALLATION_RECEIPT richiesti prima della collection')
    command = [sys.executable, '-I', '-B', str(project_root/'scripts/diagnostics/run-a001-fase0-uv/check_preflight.py'),
               '--repo', str(project_root), '--source', source, '--receipt', receipt]
    environment_mode = os.environ.get('RUN_TEST_MODE', 'official')
    if environment_mode not in ('official', 'standalone'):
        raise pytest.UsageError('RUN_TEST_MODE deve essere official o standalone')
    selected_mode = config.getoption('--test-mode')
    if selected_mode and 'RUN_TEST_MODE' in os.environ and selected_mode != environment_mode:
        raise pytest.UsageError('Modalità --test-mode e RUN_TEST_MODE discordanti')
    if (selected_mode or environment_mode) == 'standalone':
        command.append('--standalone')
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, close_fds=True)
    if result.returncode:
        raise pytest.UsageError('Preflight installazione fallito: '+result.stderr)


def pytest_unconfigure(config):
    import socket
    for name, original in _guard_originals.items():
        setattr(socket.socket, name, original)
    _guard_originals.clear()
