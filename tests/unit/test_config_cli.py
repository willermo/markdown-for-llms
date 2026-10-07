"""CLI reali nello stesso interprete già installato: nessuna preparazione implicita."""
import json
from pathlib import Path
import subprocess
import sys
import pytest


def invoke(workspace, *args):
    return subprocess.run([sys.executable,'-I','-B','-m','config',*args],cwd=workspace,
                          capture_output=True,text=True,timeout=20,close_fds=True)


def test_create_show_and_direct_script_equivalent(temp_workspace):
    made=invoke(temp_workspace,'--create-default')
    assert made.returncode == 0, made.stderr
    expected=json.loads((temp_workspace/'pipeline_config.json').read_text())
    shown=invoke(temp_workspace,'--show')
    assert shown.returncode == 0
    assert json.loads(shown.stdout) == expected
    root=Path(__file__).resolve().parents[2]
    direct=subprocess.run([sys.executable,'-B',str(root/'config.py'),'--show'],cwd=temp_workspace,
                          capture_output=True,text=True,timeout=20,close_fds=True)
    assert direct.returncode == 0 and json.loads(direct.stdout)==expected


def test_console_entrypoint_uses_same_environment(temp_workspace):
    console=Path(sys.executable).parent/'markdown-config'
    result=subprocess.run([str(console),'--show'],cwd=temp_workspace,capture_output=True,text=True,timeout=20,close_fds=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)==json.loads(invoke(temp_workspace,'--show').stdout)


def test_help_and_bad_option(temp_workspace):
    result=invoke(temp_workspace,'--help')
    assert result.returncode == 0 and all(x in result.stdout for x in ['--show','--validate','--create-default'])
    assert invoke(temp_workspace,'--unknown-option').returncode == 2


def test_config_only_does_not_construct_pipeline(temp_workspace):
    result=subprocess.run([sys.executable,'-I','-B','-m','master_workflow','--config-only'],
                          cwd=temp_workspace,capture_output=True,text=True,timeout=20,close_fds=True)
    assert result.returncode == 0, result.stderr
    assert (temp_workspace/'pipeline_config.json').is_file()
    assert not (temp_workspace/'pipeline_logs/pipeline_state.json').exists()
