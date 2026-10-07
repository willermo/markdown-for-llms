"""Confini di lancio: fasi fake esplicite e preflight isolato reale."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from master_workflow import PipelineOrchestrator
from config import ConfigManager


def test_phase_argv_and_environment(temp_workspace,monkeypatch):
    monkeypatch.chdir(temp_workspace)
    monkeypatch.setenv('PYTHONPATH',str(temp_workspace))
    monkeypatch.setenv('PYTHONHOME',str(temp_workspace))
    manager=ConfigManager();manager.config.skip_existing=False
    p=PipelineOrchestrator(manager);p.setup_directories()
    for kind in ['converted','cleaned','validated']:
        (manager.get_directory_path(kind)/'sample.md').write_text('# Titolo\n\nTesto sintetico.')
    with patch('master_workflow.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')) as run:
        assert p.run_cleaning() and p.run_validation() and p.run_chunking()
    commands=run.call_args_list
    assert len(commands)==3
    for call,module in zip(commands,['clean_markdown','validate_markdown','chunk_markdown']):
        assert call.args[0][:5]==[sys.executable,'-I','-B','-m',module]
        assert call.kwargs['cwd']==temp_workspace.resolve()
        assert not {'PYTHONPATH','PYTHONHOME'} & call.kwargs['env'].keys()
    assert '--input-dir' in commands[-1].args[0]


def test_missing_installed_module_is_explicit(temp_workspace,monkeypatch):
    monkeypatch.chdir(temp_workspace)
    p=PipelineOrchestrator(ConfigManager())
    with patch('master_workflow.subprocess.run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='Modulo installato mancante')) as run:
        assert not p.check_prerequisites()
    assert run.call_args.args[0][:4]==[sys.executable,'-I','-B','-c']


def test_real_child_spec_ignores_workspace_and_pythonpath(temp_workspace):
    marker=temp_workspace/'executed'
    (temp_workspace/'clean_markdown.py').write_text("raise RuntimeError('sentinel executed')")
    code="import importlib.util,json;print(json.dumps(importlib.util.find_spec('clean_markdown').origin))"
    env=dict(os.environ,PYTHONPATH=str(temp_workspace));env.pop('PYTHONHOME',None)
    r=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=temp_workspace,env=env,
                     capture_output=True,text=True,timeout=20,close_fds=True)
    assert r.returncode==0,r.stderr
    assert 'site-packages' in r.stdout and str(temp_workspace) not in r.stdout
    assert not marker.exists()


def test_local_socketpair_and_network_guard():
    import socket
    import pytest
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        with pytest.raises(RuntimeError,match='vietati'):s.connect(('192.0.2.1',9))
        with pytest.raises(RuntimeError):s.connect_ex(('192.0.2.1',9))
        with pytest.raises(RuntimeError):s.sendto(b'x',('192.0.2.1',9))
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        with pytest.raises(RuntimeError):s.connect('/run/docker.sock')
    a,b=socket.socketpair()
    with a,b:
        a.sendall(b'local');assert b.recv(5)==b'local'
