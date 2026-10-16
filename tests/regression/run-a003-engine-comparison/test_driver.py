"""Contratti a003 su dati sintetici; mock, non conversioni o OCR reali."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import contextlib
import shutil
import uuid

ROOT = Path(__file__).absolute().parents[3]
DRIVER = ROOT/'scripts/diagnostics/run-a003-fase1-engine-comparison'


def module(name):
    spec=importlib.util.spec_from_file_location('a003_test_'+name,DRIVER/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


p=module('text_projection')
d=module('driver')
c=module('confinement')

@contextlib.contextmanager
def attempt_tmp():
    parent = c.RUN/'work'/c.DELIVERY/'attempts'
    parent.mkdir(parents=True,exist_ok=True)
    leaf = parent/('unit-'+uuid.uuid4().hex)
    leaf.mkdir()
    try: yield str(leaf)
    finally:
        shutil.rmtree(leaf)
        shutil.rmtree(c.RUN/'evidence'/c.DELIVERY/'attempts'/leaf.name,ignore_errors=True)

def attempt_evidence(manifest, scratch):
    e = c.RUN/'evidence'/c.DELIVERY/'attempts'/scratch.name
    e.mkdir(parents=True,exist_ok=True)
    config=scratch/'consumer.json'
    if not config.exists(): config.write_text('{}')
    manifest.update(process_output=str(e/'consumer-process'),result=str(e/'consumer-result.json'),staging={'consumer.json':c.descriptor(config)},readonly_leaf=[str(config)])


def annotation(raw, specs):
    segments=[];cursor=0
    for text,op,emit,attrs in specs:
        end=cursor+len(text.encode())
        segments.append(dict(span=[cursor,end],raw=text,operation=op,emit=emit,unit_id='u',classification='CONTENT' if op=='KEEP' else 'SYNTAX',**attrs))
        cursor=end
    a=dict(version=p.VERSION,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),segments=segments,contexts=[],relations={})
    for seg in segments:
        if seg['operation']=='T02':a['contexts'].append(dict(kind='prose',span=seg['span'],unit_id=seg['unit_id']))
    grouped={}
    for seg in segments:
        if seg['operation']=='T04':grouped.setdefault(seg['relation'],[]).append(seg)
    for ident,group in grouped.items():
        if len(group)==4 and group[0]['raw']=='[':
            span=[group[0]['span'][1],group[1]['span'][0]]
            a['relations'][ident]=dict(kind='link',channel='references',segments=[g['span'] for g in group],label_span=span,label=raw[slice(*span)].decode(),target=group[2]['raw'])
    return a


class Projection(unittest.TestCase):
    def project(self,raw,specs):return p.project(raw.encode(),annotation(raw.encode(),specs))['units']['u']
    def test_TX01_escape(self):
        raw='costo \\$5';actual=self.project(raw,[('costo ','KEEP','costo ',{}),('\\$','T02','$',{'domain':'prose'}),('5','KEEP','5',{})])
        self.assertEqual(actual,'costo $5');self.assertEqual(p.measure('costo $5',actual)['CER']['distance'],0)
    def test_TX02_negation(self):
        a={'domain':'prose','pair_id':'e'}
        actual=self.project('**no**',[('**','T03','',dict(a,side='open')),('no','KEEP','no',{}),('**','T03','',dict(a,side='close'))])
        self.assertEqual(p.measure('non',actual)['CER']['distance'],1)
    def test_TX03_break(self):
        self.assertEqual(self.project('A\nB',[('A','KEEP','A',{}),('\n','T00',' ',{'domain':'prose','mode':'soft-wrap'}),('B','KEEP','B',{})]),'A B')
        self.assertEqual(p.measure('A B','A\nB')['CER']['distance'],1)
    def test_TX04_TX05_HTML_content(self):
        for raw,open_tag,content,close_tag in [('<sup>1</sup>','<sup>','1','</sup>'),('<span id="x">non</span>','<span id="x">','non','</span>')]:
            actual=self.project(raw,[(open_tag,'T06','',{'evidence':'synthetic passive tag'}),(content,'KEEP',content,{}),(close_tag,'T06','',{'evidence':'synthetic passive tag'})]);self.assertEqual(actual,content)
    def test_TX06_targets_not_prose(self):
        raw='Vedi [fonte](https://example.invalid/a_b?x=0012)'
        relation={'relation':'l1','evidence':'explicit finite link annotation'}
        actual=self.project(raw,[('Vedi ','KEEP','Vedi ',{}),('[','T04','',dict(relation,role='delimiter')),('fonte','KEEP','fonte',{}),('](','T04','',dict(relation,role='delimiter')),('https://example.invalid/a_b?x=0012','T04','',dict(relation,role='destination')),(')','T04','',dict(relation,role='delimiter'))]);self.assertEqual(actual,'Vedi fonte')
    def test_TX07_missing_visible_note(self):
        with self.assertRaises(ValueError):self.project('Vedi[^1]',[('Vedi','KEEP','Vedi',{}),('[^1]','T05','',{'relation':'note1','evidence':'synthetic call'})])
    def test_TX08_numbers(self):
        self.assertEqual(p.measure('0012','12')['CER']['distance'],2);self.assertEqual(p.measure('−7','-7')['CER']['distance'],1)
    def test_TX09_literal(self):
        self.assertEqual(self.project('a_b $x_i$',[('a_b $x_i$','KEEP','a_b $x_i$',{})]),'a_b $x_i$')
        with self.assertRaises(ValueError):self.project('\\$',[('\\$','T02','$',{'domain':'code'})])
    def test_TX10_active_HTML(self):
        with self.assertRaises(ValueError):self.project('<script>',[('<script>','T06','',{'evidence':'bad'})])
        raw=b'<script>non</script>si';a=annotation(raw,[(raw.decode(),'KEEP',raw.decode(),{})]);a['segments'][0]['classification']='UNRESOLVED'
        self.assertEqual(p.project(raw,a)['unresolved'],['u'])
    def test_TX11_missing_extra(self):
        result=p.assess([{'id':'a','text':'A','kind':'prose'},{'id':'b','text':'B','kind':'prose'}],[{'id':'o','text':'A','kind':'prose'},{'id':'x','text':'X','kind':'prose'}],{'a':{'status':'MATCHED','observed_ids':['o']},'b':{'status':'MISSING'}})
        self.assertEqual(result['gold_char_denominator'],2);self.assertEqual(result['units'][1]['measurement']['CER']['distance'],1);self.assertEqual(result['extras'][0]['measurement']['CER']['distance'],1)
    def test_TX12_zero(self):
        self.assertEqual(p.measure('','extra')['CER']['status'],'UNAVAILABLE_ZERO_DENOMINATOR');self.assertEqual(p.measure('','')['CER']['status'],'NOT_APPLICABLE')
    def test_TX13_pagination(self):
        raw='\n\n{0}'+'-'*48+'\n\n';self.assertEqual(self.project(raw,[(raw,'T08','',{'kind':'page','paginate':True,'page_id':0})]),'')
        self.assertEqual(self.project(raw,[(raw,'KEEP',raw,{})]),raw)
    def test_TX14_Unicode(self):self.assertEqual(p.measure('é','e\u0301')['CER']['distance'],2)
    def test_TX15_caps_and_boundaries(self):
        self.assertEqual(p.edits('a'*2000,'b'*2001)['status'],'UNAVAILABLE_CAP')
        raw='é'.encode();a=annotation(raw,[('é','KEEP','é',{})]);a['segments'][0]['span'][1]=1
        with self.assertRaises(ValueError):p.project(raw,a)
        for span in [[1,2],[0,1]]:
            a=annotation(b'ab',[('ab','KEEP','ab',{})]);a['segments'][0]['span']=span
            with self.assertRaises(ValueError):p.project(b'ab',a)
    def test_entity_and_br(self):
        with self.assertRaises(ValueError):self.project('&amp;<br>',[('&amp;','T07','&',{'domain':'html-text'}),('<br>','T06','\n',{})])
        with self.assertRaises(ValueError):self.project('&bogus;',[('&bogus;','T07','?',{'domain':'html-text'})])
    def test_fake_code_math_never_hides_prose(self):
        for kind in ['math','code','table','figure']:
            result=p.assess([{'id':'g','text':'non','kind':'prose'}],[{'id':'o','text':'','kind':kind}],{'g':{'status':'MATCHED','observed_ids':['o']}})
            self.assertEqual(result['units'][0]['measurement']['CER']['distance'],3);self.assertTrue(result['units'][0]['critical_type_loss'])
    def test_ambiguity_keeps_denominators(self):
        result=p.assess([{'id':'g','text':'abc','kind':'prose','class':'C10-native'}],[],{'g':{'status':'AMBIGUOUS'}})
        self.assertEqual(result['gold_char_denominator'],3);self.assertEqual(result['units'][0]['measurement'],'UNAVAILABLE')
    def test_tie_break(self):self.assertEqual(p.edits('a','b')['edits'][0]['op'],'substitution')
    def test_missing_empty_cell_is_not_present_empty_cell(self):
        expected=[{'id':'r0c0','text':'','kind':'cell'}]
        missing=p.assess(expected,[],{'r0c0':{'status':'MISSING'}})['units'][0]
        present=p.assess(expected,[{'id':'o','text':'','kind':'cell'}],{'r0c0':{'status':'MATCHED','observed_ids':['o']}})['units'][0]
        self.assertTrue(missing['critical_missing']);self.assertFalse(present['critical_missing'])
        self.assertEqual(missing['measurement']['CER']['status'],'NOT_APPLICABLE')
    def test_duplicated_caption_stays_extra(self):
        result=p.assess([{'id':'f-caption','text':'Figure 1','kind':'prose'}],[{'id':'caption','text':'Figure 1','kind':'prose'},{'id':'alt','text':'Figure 1','kind':'prose'}],{'f-caption':{'status':'MATCHED','observed_ids':['caption']}})
        self.assertEqual(result['extras'][0]['measurement']['CER']['distance'],8)
    def test_note_body_and_figure_domain_preserved(self):
        self.assertEqual(self.project('[^1]: non',[('[^1]: ','T01','',{'wrapper':'note','evidence':'finite note definition'}),('non','KEEP','non',{})]),'non')
        raw=b'figure payload';a=annotation(raw,[(raw.decode(),'KEEP',raw.decode(),{})]);a['segments'][0].update(classification='DOMAIN_OTHER',channel='figure',evidence='finite observed figure payload')
        before=hashlib.sha256(raw).hexdigest();self.assertEqual(p.project(raw,a)['units']['u'],'figure payload');self.assertEqual(hashlib.sha256(raw).hexdigest(),before)


class Lifecycle(unittest.TestCase):
    def test_pdf_helper_does_not_shadow_stdlib_profile(self):
        import subprocess,sys
        code="import importlib.util;from pathlib import Path;spec=importlib.util.spec_from_file_location('pdf_child',"+repr(str(DRIVER/'pdf_child.py'))+");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.safety_relative("+repr(str(ROOT/'scripts/benchmarks'))+");import cProfile,profile;assert callable(cProfile.run) and callable(profile.run)"
        child=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True,close_fds=True,timeout=30)
        self.assertEqual(child.returncode,0,child.stderr)
    def test_failures_do_not_turn_green(self):
        for receipt,present,size,probes,want in [({'adapter_exit':1},True,10,True,'ERROR'),({'adapter_exit':0,'stop_reason':'TIMEOUT'},True,10,True,'TIMEOUT'),({'adapter_exit':0},True,0,True,'EMPTY'),({'adapter_exit':0},False,0,True,'ERROR'),({'adapter_exit':0},True,10,False,'NOT_MEASURED')]:self.assertEqual(d.outcome(receipt,present,size,probes),want)
        self.assertIsNone(d.engine_exit({'adapter_exit':1},'txt-literal-v1',False))
        self.assertIsNone(d.engine_exit({'adapter_exit':2},'pandoc-legacy-v1',True))
        self.assertEqual(d.engine_exit({'adapter_exit':2},'pandoc-direct-v1',True),2)
    def test_new_only_omonimi_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ['case-a-attempt1','case-b-attempt1','case-a-attempt2']:
                p=Path(root)/name;p.mkdir();c.write_new(p/'raw.json',{'name':'same.pdf'})
            with self.assertRaises(FileExistsError):c.write_new(Path(root)/'case-a-attempt1/raw.json',{})
    def test_asset_escape_and_symlink(self):
        sys=__import__('sys');sys.path.insert(0,str(ROOT/'scripts/benchmarks'));import safety
        for name in ['../escape','https://example.invalid/x','%2e%2e/x','x\\y','/tmp/x']:
            with self.assertRaises(ValueError):safety.relative(name)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'raw.md').write_text('raw unchanged');(p/'link').symlink_to(p/'raw.md')
            with self.assertRaises(ValueError):d.artifact_inventory(p,safety)
            self.assertEqual((p/'raw.md').read_text(),'raw unchanged')
    def test_single_boundary_always_private_network_and_proc(self):
        with attempt_tmp() as root:
            p=Path(root);env={k:str(p) for k in ['HOME','TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME']}
            manifest={'schema':1,'phase':'conversion','scratch':str(p),'readonly':['/usr'],'files':[],'argv':['/usr/bin/true'],'environment':env,
                      'python':'/usr/bin/python3','consumer':'/usr/consumer.py','consumer_config':str(p/'consumer.json')}
            attempt_evidence(manifest,p)
            argv=c.argv(manifest,netns='net:[123]',synthetic_socket='/tmp/synthetic')
            for flag in ('--unshare-net','--unshare-pid','--unshare-user','--proc','--cap-drop','--clearenv'):
                self.assertIn(flag,argv)
            self.assertIn('A003_HOST_NETNS',argv);self.assertNotIn('A003_EXPECTED_NETNS',argv)
            self.assertEqual(argv[argv.index('--proc')+1],'/proc')
            self.assertIn('A003_SYNTHETIC_SOCKET',argv)
    def test_runner_rejects_runtime_outside_target_before_launch(self):
        import json
        from types import SimpleNamespace
        v=__import__('importlib.util',fromlist=['']);spec=v.spec_from_file_location('a003_runner',ROOT/'scripts/diagnostics/run-a001-fase0-uv/run_offline.py');runner=v.module_from_spec(spec);spec.loader.exec_module(runner)
        with attempt_tmp() as root:
            p=Path(root);cfg=p/'consumer.json';cfg.write_text(json.dumps({'runtime':{'python':'wrong'},'argv':['/usr/bin/true']}))
            env={k:str(p) for k in ['HOME','TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME']}
            manifest={'schema':1,'phase':'conversion','scratch':str(p),'readonly':['/usr'],'files':[],'argv':['/usr/bin/true'],'environment':env,'python':'/usr/bin/python3','consumer':str(DRIVER/'consumer.py'),'consumer_config':str(cfg)}
            attempt_evidence(manifest,p)
            mf=Path(manifest['process_output']).parent/'manifest.json';mf.write_text(json.dumps(manifest))
            args=SimpleNamespace(bundle=None,inside=False,command=[],confinement=mf,preflight_config=p/'preflight.json',repo=ROOT,python='/usr/bin/python3',cwd=p)
            target={k:[] for k in ('dependencies','startup')};target.update(python='/usr/bin/python3',prefix='/usr',base_prefix='/usr',python_version='3.12.13')
            with self.assertRaisesRegex(ValueError,'Runtime consumer'):runner.confined_command(args,target,None,'/tmp/s')
    def test_driver_docling_closure_gate_requires_manifest(self):
        v=__import__('importlib.util',fromlist=['']);spec=v.spec_from_file_location('a003_verify',ROOT/'scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py');verify=v.module_from_spec(spec);spec.loader.exec_module(verify)
        with self.assertRaisesRegex(ValueError,'manifest closure'):verify.check_docling_closure({},[],None)
        self.assertEqual(verify.docling_metadata_member(['setuptools-84.0.0.dist-info/METADATA','setuptools/_vendor/packaging-26.0.dist-info/METADATA']),'setuptools-84.0.0.dist-info/METADATA')
        with self.assertRaises(ValueError):verify.docling_metadata_member(['x.dist-info/METADATA','y.dist-info/METADATA'])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'closure.json';path.write_text('{"schema":1}')
            receipt={'profile':'docling','installed':{'docling_closure':{'manifest':verify.descriptor(path)}}}
            self.assertEqual(verify.receipt_closure_path(receipt),path)
            path.write_text('{"schema":2}')
            with self.assertRaisesRegex(ValueError,'closure Docling cambiato'):verify.receipt_closure_path(receipt)
            self.assertIsNone(verify.receipt_closure_path({'profile':'cpu'}))


if __name__=='__main__':unittest.main()
