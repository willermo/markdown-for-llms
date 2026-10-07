"""Negativi di ambiente/binding/startup: stdlib, nessun import applicativo."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/diagnostics/run-a001-fase0-uv'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class CoreBoundaries(unittest.TestCase):
    def test_closed_environment_never_merges_secrets(self):
        tool=load('run_offline')
        env={k:'public' for k in ['PATH','HOME','LANG','LC_ALL','TMPDIR','TIKTOKEN_CACHE_DIR','PYTHONNOUSERSITE',
                                 'PIP_CONFIG_FILE','TZ','UV_CACHE_DIR','UV_PYTHON_DOWNLOADS','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_CONFIG_DIRS']}
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1')
        t={'environment':env,'tmpdir':'public','tokenizer_cache':'public'}
        with patch.dict(os.environ,{'COVERAGE_PROCESS_START':'evil','MARKER_API_KEY':'secret','PYTHONPATH':'evil','https_proxy':'evil'}):
            self.assertEqual(tool.clean_env(t),env)
        t['environment']=dict(env,HTTP_PROXY='evil')
        with self.assertRaises(ValueError):tool.clean_env(t)
        t['environment']=env;t['workload_environment']={'DOCKER_HOST':'unix:///run/docker.sock'}
        with self.assertRaises(ValueError):tool.clean_env(t)

    def test_target_startup_checked_before_launch(self):
        tool=load('run_offline')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.pth';p.write_bytes(b'# benign\n')
            desc={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            target={'files':[desc],'startup':{'site_dirs':[d],'pth':[desc]}}
            tool.target_files_check(target)
            (Path(d)/'evil.pth').write_bytes(b'import evil\n')
            with self.assertRaises(ValueError):tool.target_files_check(target)
            (Path(d)/'evil.pth').unlink();p.write_bytes(b'import bad\n')
            with self.assertRaises(ValueError):tool.target_files_check(target)

    def test_missing_or_mutated_preliminary_binding_rejected(self):
        tool=load('run_offline')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'input';p.write_text('input');bundle=root/'bundle.json'
            bundle.write_text(json.dumps({'schema':1,'scope':'preliminary-inputs','files':[{'path':'input','bytes':5,'sha256':hashlib.sha256(b'input').hexdigest()}]}))
            a=SimpleNamespace(snapshot=None,bundle=bundle,repo=root,target=p)
            with self.assertRaises(ValueError):tool.binding_check(a)  # Wrapper non legato.
            p.write_text('changed')
            with self.assertRaises(ValueError):tool.binding_check(a)

    def test_baseline_argv_isolated_and_closed(self):
        tool=load('run_baseline')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(tool.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')) as run:
                tool.command([tool.sys.executable,str(root/'config.py'),'--show'],root,'synthetic',root)
            argv=run.call_args.args[0]
            self.assertEqual(argv[:3],[tool.sys.executable,'-I','-B'])
            self.assertIn('--root',argv)
            self.assertNotIn('PYTHONPATH',run.call_args.kwargs['env'])

    def test_empty_baseline_config_stops_ancestor_discovery(self):
        tool=load('run_baseline')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'baseline-pytest.ini').touch()
            argv=tool.legacy_pytest_args(root,root/'cache','collection')
            self.assertEqual(argv[argv.index('-c')+1],str(root/'baseline-pytest.ini'))


class LauncherCleanup(unittest.TestCase):
    def exercise(self, error, cleanup_fails=False):
        from unittest.mock import MagicMock
        tool=load('run_offline')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);socket_dir=root/'socket';socket_dir.mkdir();output=root/'receipt'
            listener=MagicMock();listener.__enter__.return_value=listener
            listener.bind.side_effect=lambda path:Path(path).write_bytes(b'synthetic placeholder')
            args=SimpleNamespace(output=output,runner='firejail',target=ROOT/'AGENTS.md',repo=ROOT,
                snapshot=ROOT/'unused-snapshot.json',python=sys.executable,cwd=ROOT,
                preflight_config=None,command=[],expected_exit=0,timeout=1,bundle=None)
            target=dict(daemons=[],tmpdir=str(root),socket_root=str(root),binaries={'firejail':'/synthetic/firejail'})
            original_unlink=Path.unlink
            def unlink(path,*a,**kw):
                if cleanup_fails and path==socket_dir/'s':raise PermissionError('synthetic cleanup EACCES')
                return original_unlink(path,*a,**kw)
            with patch.object(tool,'binding_check',return_value={}),patch.object(tool,'target_files_check'), \
                 patch.object(tool,'clean_env',return_value={}),patch.object(tool.tempfile,'mkdtemp',return_value=str(socket_dir)), \
                 patch.object(tool.socket,'socket',return_value=listener),patch.object(tool.os,'fsencode',return_value=b'/short/socket'), \
                 patch.object(tool.subprocess,'run',side_effect=error),patch.object(Path,'unlink',unlink):
                self.assertEqual(tool.outside(args,target),2)
            receipt=json.loads((output/'receipt.json').read_text())
            self.assertNotEqual(receipt['status'],'PASS');self.assertIn(str(error),receipt['error'])
            self.assertEqual(receipt['temporary_socket_cleaned'],not cleanup_fails)
            self.assertEqual(socket_dir.exists(),cleanup_fails)
            if cleanup_fails:self.assertIn('synthetic cleanup EACCES',receipt['cleanup_error'])
            else:self.assertNotIn('cleanup_error',receipt)
            return receipt

    def test_eacces_after_bind_keeps_original_and_collects_socket(self):
        self.exercise(PermissionError('synthetic runner EACCES'))

    def test_timeout_after_bind_keeps_original_and_collects_socket(self):
        import subprocess
        self.exercise(subprocess.TimeoutExpired(['synthetic-firejail'],1))

    def test_cleanup_failure_has_receipt_and_original_error(self):
        self.exercise(PermissionError('synthetic runner EACCES'),cleanup_fails=True)


class LauncherSnapshots(unittest.TestCase):
    """Validation mocks only; real R is exercised separately in run evidence."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name);self.tool=load('run_offline')
        self.data=dict(schema=1,run_id='run-a001-fase0-uv',label='test-s001',
                       head='a'*40,branch='synthetic',files=[],artifacts=[],worktree_sha256='b'*64)
    def snapshot(self, data=None):
        value=self.data if data is None else data
        path=self.repo/'temp'/self.data['run_id']/'snapshots'/(self.data['label']+'.json')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    def test_both_valid_run_ids_invoke_verifier_with_target_python(self):
        for run_id in ('run-a001-fase0-uv','run-b002-fase0-uv'):
            with self.subTest(run_id=run_id):
                self.data['run_id']=run_id;path=self.snapshot()
                with patch.object(self.tool.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='MATCH',stderr='')) as run:
                    result=self.tool.snapshot_check(self.repo,path,'/verified/python')
                self.assertEqual(run.call_args.args[0],['/verified/python','-I','-B',str(self.repo/'scripts/run_context.py'),'verify',run_id,'--label','test-s001'])
                self.assertEqual(result['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
    def test_invalid_schema_id_label_and_path_stop_before_verifier(self):
        for key,value in [('schema',True),('schema',2),('schema',1.0),('run_id','../escape'),('run_id','run-x'),('label','../escape'),('label','/absolute'),('label','UPPER')]:
            with self.subTest(key=key,value=value):
                path=self.snapshot(dict(self.data,**{key:value}))
                with patch.object(self.tool.subprocess,'run') as run,self.assertRaises(ValueError):
                    self.tool.snapshot_check(self.repo,path,'/verified/python')
                run.assert_not_called()
        path=self.snapshot();wrong=path.parent/'wrong.json';wrong.write_bytes(path.read_bytes())
        with patch.object(self.tool.subprocess,'run') as run,self.assertRaises(ValueError):self.tool.snapshot_check(self.repo,wrong,'/verified/python')
        run.assert_not_called()
    def test_snapshot_and_parent_symlinks_rejected_before_verifier(self):
        path=self.snapshot();original=path.parent/'original';path.rename(original);path.symlink_to(original)
        with patch.object(self.tool.subprocess,'run') as run,self.assertRaises(ValueError):self.tool.snapshot_check(self.repo,path,'/verified/python')
        run.assert_not_called();path.unlink();original.rename(path)
        parent=path.parent;parent.rename(parent.with_name('real'));parent.symlink_to(parent.with_name('real'),target_is_directory=True)
        with patch.object(self.tool.subprocess,'run') as run,self.assertRaises(ValueError):self.tool.snapshot_check(self.repo,path,'/verified/python')
        run.assert_not_called()
    def test_stale_stops_before_socket_or_workload_and_has_receipt(self):
        path=self.snapshot();target=self.repo/'target.json';target.write_text('{}');out=self.repo/'output'
        args=SimpleNamespace(snapshot=path,bundle=None,repo=self.repo,python='/verified/python',target=target,output=out,runner='firejail')
        with patch.object(self.tool.subprocess,'run',return_value=SimpleNamespace(returncode=2,stdout='STALE',stderr='')) as run,patch.object(self.tool.tempfile,'mkdtemp') as socket_dir:
            self.assertEqual(self.tool.outside(args,{}),2)
        self.assertEqual(run.call_count,1);socket_dir.assert_not_called()
        receipt=json.loads((out/'receipt.json').read_text());self.assertEqual(receipt['status'],'FAIL');self.assertIn('STALE',receipt['error']);self.assertNotIn('invocation',receipt)


if __name__=='__main__':unittest.main()
