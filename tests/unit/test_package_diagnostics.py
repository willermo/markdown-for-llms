"""Test stdlib puri: dati sintetici, nessun pytest/conftest/app/S operativo."""
import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('package_check', ROOT/'scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
source = verify.source_tool


def record_bytes(name, content):
    return {'path':name,'kind':'file','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}


def metadata_bytes():
    lines = ['Name: markdown-for-llms','Version: 1.0.0','Requires-Python: >=3.12,<3.13']
    lines += ['Requires-Dist: '+r for r in ('requests>=2.31.0','tiktoken>=0.5.1','tqdm>=4.66.0','python-dotenv>=1.0.0')]
    for extra in ('marker-server','marker-cpu','marker-cu126'):
        lines.append('Provides-Extra: '+extra)
        deps = ('fastapi','uvicorn','python-multipart') if extra == 'marker-server' else ('marker-pdf[full]==1.10.2','surya-ocr==0.17.1','torch==2.7.1','beautifulsoup4')
        lines += ['Requires-Dist: '+d+'; extra == "'+extra+'"' for d in deps]
    return ('\n'.join(lines)+'\n\n').encode()


class ArchiveIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def zip(self, entries):
        path = self.root/'synthetic.whl'
        with zipfile.ZipFile(path,'w') as f:
            for name, data in entries:
                f.writestr(name,data)
        return path

    def test_zip_traversal_absolute_and_unknown_rejected(self):
        for name in ('../config.py','/config.py','x/../config.py','./config.py','C:/config.py','test_secret.py','.env','temp/run.py'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify.read_archive(self.zip([(name,b'data')]),True)

    def test_zip_duplicate_rejected(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            path = self.zip([('config.py',b'first'),('config.py',b'second')])
        with self.assertRaises(ValueError):
            verify.read_archive(path,True)

    def test_zip_symlink_rejected(self):
        info = zipfile.ZipInfo('config.py')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            verify.read_archive(self.zip([(info,b'/etc/passwd')]),True)

    def test_tar_links_and_traversal_rejected(self):
        for name, kind in [('markdown_for_llms-1.0.0/config.py',tarfile.SYMTYPE),
                           ('markdown_for_llms-1.0.0/config.py',tarfile.LNKTYPE),
                           ('markdown_for_llms-1.0.0/../config.py',tarfile.REGTYPE)]:
            p = self.root/'synthetic.tar.gz'
            with tarfile.open(p,'w:gz') as archive:
                info = tarfile.TarInfo(name); info.type = kind; info.linkname = '/outside'
                archive.addfile(info, io.BytesIO(b''))
            with self.subTest(name=name,kind=kind), self.assertRaises(ValueError):
                verify.read_archive(p,False)

    def test_record_real_digest_and_dimensions(self):
        files = {'config.py':b'number = 123\n',verify.DIST+'/RECORD':b''}
        digest = base64.urlsafe_b64encode(hashlib.sha256(files['config.py']).digest()).rstrip(b'=').decode()
        valid = ('config.py,sha256='+digest+',13\n'+verify.DIST+'/RECORD,,\n').encode()
        verify.check_record(valid,files)
        for bad in (valid.replace(b',13',b',12'),valid.replace(digest.encode(),b'0'*43),
                    valid+valid,valid.splitlines(keepends=True)[1],valid.replace(b'sha256=',b'md5=')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify.check_record(bad,files)

    def test_installed_record_exceptions_do_not_cover_modules(self):
        files = {'config.py':b'x',verify.DIST+'/RECORD':b''}
        with self.assertRaises(ValueError):
            verify.check_record(b'config.py,,\n'+(verify.DIST+'/RECORD,,\n').encode(),files,True)

    def test_metadata_requires_five_exact_entrypoints(self):
        metadata = metadata_bytes()
        ep = ('[console_scripts]\n'+'\n'.join(k+' = '+v for k,v in verify.ENTRYPOINTS.items())).encode()
        verify.check_metadata(metadata,ep)
        for data, entries in ((metadata.replace(b'1.0.0',b'2.0.0'),ep),
                              (metadata,ep.replace(b'config:main',b'config:missing')),
                              (metadata,ep+b'\nextra = config:main')):
            with self.assertRaises(ValueError):
                verify.check_metadata(data,entries)

    def test_full_synthetic_chain_and_stale_module(self):
        files = {name:('synthetic '+name+'\n').encode() for name in verify.MODULE_FILES}
        for name,data in files.items():
            (self.root/name).write_bytes(data)
        metadata = metadata_bytes()
        ep = ('[console_scripts]\n'+'\n'.join(k+' = '+v for k,v in verify.ENTRYPOINTS.items())).encode()
        inputs = {'pyproject.toml':b'', 'README.md':b'readme', 'LICENSE':b'MIT',
                  '.python-version':b'3.12.13\n','build-constraints.txt':b'setuptools==84.0.0\n','MANIFEST.in':b'include config.py\n'}
        tarfiles = dict(files, **inputs, **{'PKG-INFO':metadata,verify.EGG+'/entry_points.txt':ep})
        for name,data in inputs.items():
            (self.root/name).write_bytes(data)
        # Solo descriptor in memoria per il gate unitario: nessun lock prodotto.
        inputs['uv.lock'] = b''
        zipfiles = dict(files)
        zipfiles.update({verify.DIST+'/METADATA':metadata, verify.DIST+'/entry_points.txt':ep,
                         verify.DIST+'/WHEEL':b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
                         verify.DIST+'/licenses/LICENSE':b'MIT',verify.DIST+'/top_level.txt':b''})
        rows = io.StringIO(); writer = csv.writer(rows)
        for name,data in zipfiles.items():
            writer.writerow([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(),len(data)])
        writer.writerow([verify.DIST+'/RECORD','',''])
        zipfiles[verify.DIST+'/RECORD'] = rows.getvalue().encode()
        tarpath = self.root/'synthetic.tar.gz'
        with tarfile.open(tarpath,'w:gz') as f:
            for name,data in tarfiles.items():
                info = tarfile.TarInfo('markdown_for_llms-1.0.0/'+name); info.size = len(data)
                f.addfile(info,io.BytesIO(data))
        wheel = self.zip(zipfiles.items())
        payload = {'host_repo':str(self.root),'modules':[record_bytes(n,d) for n,d in files.items()],
                   'build_inputs':[record_bytes(n,d) for n,d in inputs.items()]}
        self.assertEqual(verify.check_archives(payload,tarpath,wheel),zipfiles)
        (self.root/'exceptions.py').write_bytes(b'changed\n')
        with self.assertRaises(ValueError):
            verify.check_archives(payload,tarpath,wheel)


class SourceFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in source.MODULES:
            (self.root/(name+'.py')).write_text('synthetic\n')
        self.git = {'branch':'test','head':'a','dev':'a','merge_base':'a','status':''}
        self.payload = {'scope':'standalone','repo':str(self.root),'host_repo':str(self.root),'git':self.git,
                        'modules':sorted([source.entry(self.root/(m+'.py'),self.root) for m in source.MODULES],key=lambda x:x['path']),
                        'build_inputs':[source.entry(self.root/n,self.root) for n in (*source.INPUTS,'src')],
                        'diagnostics':[], 'input_inventory':None,'backend':{'build_system':None},
                        'toolchain':{},'snapshot':None}
        self.mockgit = patch.object(source,'git_identity',return_value=self.git)  # synthetic Git fixture only
        self.mockgit.start(); self.addCleanup(self.mockgit.stop)

    def test_standalone_has_no_temp_dependency(self):
        source.validate_current(source.envelope(self.payload),self.root)
        self.assertFalse((self.root/'temp').exists())

    def test_old_source_new_clone_fails(self):
        old = source.envelope(self.payload)
        (self.root/'exceptions.py').write_text('new source\n')
        with self.assertRaises(ValueError):
            source.validate_current(old,self.root)

    def test_correct_id_wrong_bytes_or_hash_fails(self):
        for field,value in [('bytes',999),('sha256','0'*64)]:
            p = copy.deepcopy(self.payload); p['modules'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                source.validate_current(source.envelope(p),self.root)

    def test_missing_duplicate_and_unknown_schema_fails(self):
        for change in ('missing','duplicate','schema','unknown'):
            p = copy.deepcopy(self.payload)
            if change == 'missing': p['modules'].pop()
            if change == 'duplicate': p['modules'].append(p['modules'][0])
            if change == 'unknown': p['unknown'] = True
            value = source.envelope(p)
            if change == 'schema': value['schema'] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                source.validate_current(value,self.root)

    def test_timestamp_excluded_payload_not_excluded(self):
        value = source.envelope(self.payload); original = value['id']; value['captured_at_utc'] = 'later'
        source.check_envelope(value)
        value['payload']['git']['head'] = 'different'
        with self.assertRaises(ValueError): source.check_envelope(value)
        self.assertEqual(original,value['id'])

    def test_standalone_not_official(self):
        with self.assertRaises(ValueError): source.validate_current(source.envelope(self.payload),self.root,official=True)

    def test_paths_symlinks_duplicate_json_and_missing_input(self):
        for name in ('../config.py','/absolute','a/../b','a\\b','./a'):
            with self.subTest(name=name), self.assertRaises(ValueError): source.relative_path(name)
        (self.root/'link').symlink_to(self.root/'config.py')
        with self.assertRaises(ValueError): source.entry(self.root/'link',self.root)
        (self.root/'duplicate.json').write_text('{"schema":1,"schema":2}')
        with self.assertRaises(ValueError): source.load_json(self.root/'duplicate.json')
        (self.root/'config.py').unlink()
        with self.assertRaises(ValueError): source.validate_current(source.envelope(self.payload),self.root)

    def test_snapshot_mapping_missing_duplicate_hash(self):
        item = self.payload['modules'][0]
        source.match_snapshot([item],{item['path']:item})
        for records,index in (([item],{}),([item,item],{item['path']:item}),
                              ([item],{item['path']:dict(item,sha256='0'*64)})):
            with self.assertRaises(ValueError): source.match_snapshot(records,index)

    def test_old_receipt_new_source_and_external_sha_fail_before_origin(self):
        s = self.root/'S-synthetic.json'; s.write_text(json.dumps(source.envelope(self.payload)))
        receipt = {'kind':'I','scope':'standalone','source':dict(verify.descriptor(s),id=source.load_json(s)['id']),
                   'snapshot':None,'archives':[],'interpreter':{},'profile':'base','inputs':[],
                   'argv':[],'cwd':str(self.root),'status':'PASS','installed':{}}
        p = self.root/'I-synthetic.json'
        for mode in ('old-source','sha'):
            r = copy.deepcopy(receipt)
            if mode == 'old-source': r['source']['id'] = 'sha256:'+'0'*64
            else: r['source']['sha256'] = '0'*64
            p.write_text(json.dumps(source.envelope(r)))
            with patch.object(verify.origin_tool,'check_origin',side_effect=AssertionError('origin reached')):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    verify.validate_receipt(p,s,self.root,official=False)


class OriginContractTests(unittest.TestCase):
    def test_missing_or_relative_managed_fails(self):
        for path in (Path('relative'),Path('/nonexistent-a001-synthetic-managed')):
            with self.assertRaises(ValueError): verify.origin_tool.check_origin(path)


if __name__ == '__main__':
    unittest.main()
