"""Residui r004: matrice finita, unit/mock; nessuna inferenza o R autentico."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).absolute().parents[3]
H=ROOT/'scripts/diagnostics/run-a003-fase1-engine-comparison'
def load(path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
old=load(Path(__file__).with_name('test_contract_r004.py'))
p=old.p;c=old.c;d=old.d
annotation=old.annotation;partition=old.partition;result=old.result
VARIANTS=('whole','renamed','separate','keep-codepoints')
def case(name,specs,accepted=False):
    raw,a=annotation(specs);a['contexts']=[dict(kind='prose',span=[0,len(raw)])]
    return dict(name=name,raw=raw,annotation=a,accepted=accepted)
def escape_body():return [('x','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_')),('i','KEEP',{})]
def emphasis(prefix,marker,body,suffix):
    return [(prefix,'KEEP',{}),(marker,'T03',dict(domain='prose',pair_id='p',side='open')),(body,'KEEP',{}),(marker,'T03',dict(domain='prose',pair_id='p',side='close')),(suffix,'KEEP',{})]
def finite_cases():
    rows=[]
    for marker in ('```','````','~~~','~~~~'):
        for info in ('','text'):
            for close in (marker,marker+marker[0],marker[:-1],'a '+marker, ''):
                opening=marker+info+'\n'
                rows.append(case('fence-payload/'+repr((marker,info,close)),[(opening+'a '+marker+' b','KEEP',{}),*escape_body(),('\n'+close+'\n','KEEP',{})]))
            rows.append(case('external-prose/'+marker+info,[(marker+info+'\nx\\_i\n'+marker+'\nprosa ','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_')),(' ok','KEEP',{})],True))
            rows.append(case('external-wrapper/'+marker+info,[(marker+info+'\n','T01',dict(wrapper='fence',evidence='raw exterior')),(('a '+marker+' b\\_c\n'),'KEEP',{}),(marker+'\n','T01',dict(wrapper='fence',evidence='raw exterior'))],True))
        rows.append(case('incomplete-opening/'+marker,[(marker+'text','KEEP',{}),*escape_body()]))
    bodies=[('T02',escape_body()),('T03',emphasis('','_','x','')),
            ('T06',[('<b>','T06',dict(evidence='finite tag')),('non','KEEP',{}),('</b>','T06',dict(evidence='finite tag'))]),
            ('T00',[('x','KEEP',{}),('\n','T00',dict(mode='soft-wrap',domain='prose',emit=' ')),('i','KEEP',{})]),
            ('T01',[('# ','T01',dict(wrapper='heading',evidence='false raw wrapper')),('non','KEEP',{})]),
            ('T08',[('x','KEEP',{}),('\n\n','T08',dict(kind='blank-layout',evidence='false blank')),('i','KEEP',{})])]
    for before in ('costa 5 $; ','costa $5$; ','$$prima$$ ', '$prima `x` '):
        for delimiter in ('`','``','```'):
            for op,body in bodies:
                rows.append(case('family-union/'+repr((before,delimiter,op)),[(before+delimiter+'echo $PATH ','KEEP',{}),*body,(delimiter+' fine','KEEP',{})]))
    for marker in ('*','_','**','__'):
        for prefix,body,suffix,ok in [('a','"b"','c',False),('a','.b.','c',False),('','b','',True),('(','b',')',True),('',' b','',False),('','b ','',False),('a','b','c',marker.startswith('*'))]:
            rows.append(case('flanking/'+repr((marker,prefix,body,suffix)),emphasis(prefix,marker,body,suffix),ok))
        for n in range(1,5):
            rows.append(case('escape-parity/'+marker+'/'+str(n),emphasis('\\'*n,marker,'x',''),n%2==0))
            rows.append(case('close-escape/'+marker+'/'+str(n),emphasis('',marker,'x'+'\\'*n,''),n%2==0))
    for prefix in ('    ','\t',' \t','     '):
        for pre in ('','prosa\n\n'):
            for op,body in bodies:
                rows.append(case('indented/'+repr((prefix,pre,op)),[(pre+prefix,'KEEP',{}),*body,('\n','KEEP',{})]))
        rows.append(case('after-indented/'+repr(prefix),[(prefix+'x\\_i\nprosa ','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_'))],True))
    for schema in ('HTTPS://','Http://','MAILTO:','CuStOm+v1://'):
        rows.append(case('uri/'+schema,[(schema+'E.INVALID/','KEEP',{}),*escape_body()]))
    for before,wrapper,value in [('Il totale è ','list','12. '),('Testo ','heading','# '),('non ','quote','> '),('x ','note','[^n]: '),('x ','caption',': ')]:
        rows.append(case('midline/'+wrapper,[(before,'KEEP',{}),(value,'T01',dict(wrapper=wrapper,evidence='not position')),('non -7 0012','KEEP',{})]))
        rows.append(case('line-wrapper/'+wrapper,[('\n','KEEP',{}),(value,'T01',dict(wrapper=wrapper,evidence='raw line start')),('non -7 0012','KEEP',{})],True))
    for separator in ('\n','\r\n','\n\n','\r\n\r\n'):
        rows.append(case('layout/'+repr(separator),[('frammen','KEEP',{}),(separator,'T08',dict(kind='blank-layout',evidence='raw separator')),('to','KEEP',{})],separator.count('\n')==2))
    rows.append(case('terminal-newline',[('non -7 0012','KEEP',{}),('\n','T08',dict(kind='blank-layout',evidence='terminal'))],True))
    rows.append(case('nested',[('*','T03',dict(domain='prose',pair_id='a',side='open')),('non ','KEEP',{}),('__','T03',dict(domain='prose',pair_id='b',side='open')),('−7','KEEP',{}),('__','T03',dict(domain='prose',pair_id='b',side='close')),(' si','KEEP',{}),('*','T03',dict(domain='prose',pair_id='a',side='close'))],True))
    rows.append(case('unqualified-fence-info',[('```text `invalid`\n','T01',dict(wrapper='fence',evidence='ambiguous info')),('non\\_x\n','KEEP',{}),('```\n','T01',dict(wrapper='fence',evidence='raw closing'))]))
    return [dict(row,name=row['name']+'/'+v,annotation=partition(row['annotation'],v)) for row in rows for v in VARIANTS]

class ResidualContract(unittest.TestCase):
    def test_finite_interactions_and_partitions(self):
        for row in finite_cases():
            with self.subTest(name=row['name']):
                actual=result(row['raw'],row['annotation']);self.assertEqual(actual['accepted'],row['accepted'],actual)
                if not row['accepted']:
                    raw,a=annotation([(row['raw'].decode(),'KEEP',{})]);self.assertEqual(result(raw,a)['emission'],raw.decode())
    def test_metadata_anchor_boundaries(self):
        for before,after,ok in [('','',True),('prosa ',' non',True),('frammen','to',False)]:
            row=case('anchor',[(before,'KEEP',{}),('[]{#a}','T08',dict(kind='empty-anchor',evidence='raw empty anchor')),(after,'KEEP',{})]);self.assertEqual(result(row['raw'],row['annotation'])['accepted'],bool(ok))

class ExclusiveAttempt(unittest.TestCase):
    def test_historical_leaves_and_new_staging(self):
        for delivery,name in [('implementation-r002','a003-r002-c01-txt-001'),('implementation-r002','a003-r002-marker-c06-whole-002'),('implementation-r001','a003-c01-txt-r001'),('implementation-r003','a003-r003-guard-txt-002')]:
            path=c.RUN/'evidence'/delivery/'attempts'/name/'manifest.json';m=json.loads(path.read_text())
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'RW requires'):c.validate(m)
        with tempfile.TemporaryDirectory() as td,mock.patch.object(c,'RUN',Path(td)/'run'):
            scratch=c.RUN/'work'/c.DELIVERY/'attempts/new-fixture';scratch.mkdir(parents=True)
            m=old.manifest(scratch);self.assertEqual(c.validate(m),scratch)
            for area,name in [('scratch','document.md'),('evidence','final.json'),('evidence','consumer-result.json')]:
                path=(scratch if area=='scratch' else Path(m['result']).parent)/name;path.write_text('preserved')
                with self.subTest(area=area,name=name),self.assertRaises(ValueError):c.validate(m)
                self.assertEqual(path.read_text(),'preserved');path.unlink()
            m['staging']['document.md']=dict(path=str(scratch/'document.md'),bytes=0,sha256='fake')
            with self.assertRaises(ValueError):c.validate(m)
    def test_changed_staging_and_probe_history(self):
        with tempfile.TemporaryDirectory() as td,mock.patch.object(c,'RUN',Path(td)/'run'):
            scratch=c.RUN/'work'/c.DELIVERY/'attempts/new-fixture';scratch.mkdir(parents=True)
            m=old.manifest(scratch);(scratch/'consumer.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'STALE'):c.validate(m)
            m['staging']['consumer.json']=c.descriptor(scratch/'consumer.json')
            e=Path(m['result']).parent;(e/'R').mkdir();(e/'R/receipt.json').write_text('historical')
            with self.assertRaisesRegex(ValueError,'already used'):c.validate(m)

class CompleteFinal(unittest.TestCase):
    def test_runner_runtimeerror_cause_survives(self):
        with tempfile.TemporaryDirectory() as td:
            kw=old.PostRunContract().fixture(Path(td))
            (kw['evidence']/'R/receipt.json').write_text(json.dumps(dict(status='FAIL',stop_reason='HOST_RUN_ERROR',error='mock monitor failure',children_reaped=None)))
            value=d.finalize_attempt(**kw)
            self.assertEqual(value['operational_status'],'ERROR');self.assertEqual(value['stop_reason'],'HOST_RUN_ERROR')
            self.assertEqual(value['runner_host_error'],'mock monitor failure');self.assertIsNone(value['runner_children_reaped'])
    def test_unreadable_assets(self):
        with tempfile.TemporaryDirectory() as td:
            kw=old.PostRunContract().fixture(Path(td));assets=kw['scratch']/'assets';assets.mkdir();(assets/'figure.png').write_bytes(b'partial asset');assets.chmod(0)
            try:
                value=d.finalize_attempt(**kw)
                self.assertEqual(value['operational_status'],'ERROR');self.assertTrue(any(x['phase']=='inventory-traversal' for x in value['output_read_errors']));self.assertEqual((kw['evidence']/'raw/document.md').read_bytes(),b'raw')
            finally:assets.chmod(0o700)
    def test_regular_sentinel_bytes_and_identity_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            kw=old.PostRunContract().fixture(Path(td));leaf=kw['scratch']/'outside-link';leaf.unlink();leaf.write_bytes(b'engine partial\x00\xff');info=leaf.lstat()
            value=d.finalize_attempt(**kw);self.assertEqual(value['operational_status'],'ERROR')
            self.assertEqual(leaf.read_bytes(),b'engine partial\x00\xff');self.assertEqual(leaf.lstat().st_ino,info.st_ino)
            self.assertEqual(Path(value['sentinel_replacement']['reading_copy']).read_bytes(),leaf.read_bytes())
    def test_runtime_error_on_started_attempt(self):
        stage=json.loads((c.RUN/'evidence/implementation-r004/preflight-r007/stage-base.json').read_text());stage['delivery']=c.DELIVERY
        with tempfile.TemporaryDirectory(dir=c.RUN/'work'/c.DELIVERY,prefix='fixture-') as td,mock.patch.object(d.c,'RUN',Path(td)/'run'):
            # Il native usa il root operativo del modulo, sostituito solo in questa fixture.
            denied=d.c.RUN/'work'/c.DELIVERY/'synthetic-denied';denied.mkdir(parents=True);(denied/'gold').write_text('synthetic')
            def failed(*args,**kwargs):
                scratch=d.c.RUN/'work'/c.DELIVERY/'attempts/mock-runtime';(scratch/'document.md').write_bytes(b'partial')
                raise RuntimeError('private monitor failed; cleanup not proven')
            monitor=load(ROOT/'scripts/benchmarks/resources.py')
            # L'import standard resources in native usa lo stesso modulo dal sys.modules.
            import sys
            with mock.patch.dict(sys.modules,{'resources':monitor}),mock.patch.object(monitor,'run',side_effect=failed):
                value=d.native(ROOT,stage,'C01-txt-it','txt-literal-v1','mock-runtime')
            self.assertEqual(value['stop_reason'],'HOST_RUN_ERROR');self.assertEqual(value['operational_status'],'ERROR');self.assertFalse(value['children_reaped']);self.assertEqual(value['raw']['bytes'],7)
            self.assertTrue(any('private monitor failed' in x['error'] for x in value['output_read_errors']))

class ExtraNumerators(unittest.TestCase):
    def sample(self,extra=' ',right_class=None):
        cls=dict(format='txt',page=1,native=True,language='it');other=right_class or cls
        gold=[dict(id='a',text='non',kind='prose',**{'class':cls}),dict(id='b',text='0012',kind='prose',**{'class':other})]
        raw=('non'+extra+'0012').encode();b=3+len(extra.encode())
        obs=[dict(id='a',text='non',kind='prose',raw_spans=[[0,3]]),dict(id='b',text='0012',kind='prose',raw_spans=[[b,len(raw)]])]
        if extra:obs.append(dict(id='x',text=extra,kind='prose',raw_spans=[[3,b]]))
        return p.assess(gold,obs,{x['id']:dict(status='MATCHED',observed_ids=[x['id']]) for x in gold},raw=raw)
    def test_native_zero_extra_and_denominator(self):
        assessment=self.sample();v=p.text_aggregates(assessment)[0]['metrics'];self.assertEqual((v['CER']['distance'],v['CER']['n']),(1,7));self.assertEqual(v['WER']['distance'],0);self.assertEqual(p.native_text_verdict(assessment),'FAIL')
        self.assertEqual(assessment['extras'][0]['measurement']['CER']['status'],'UNAVAILABLE_ZERO_DENOMINATOR');self.assertEqual(assessment['extras'][0]['status'],'FAIL_EXTRA')
    def test_no_extra_positive_and_ambiguity(self):
        self.assertEqual(p.native_text_verdict(self.sample('')),'PASS')
        value=self.sample(right_class=dict(format='txt',page=2,native=True,language='it'))
        self.assertTrue(all(x['metrics']['CER']['status']=='UNAVAILABLE' for x in p.text_aggregates(value)))
        self.assertEqual(p.native_text_verdict(value),'UNAVAILABLE')
    def test_missing_locators_never_skip(self):
        value=p.assess([dict(id='g',text='non',kind='prose',**{'class':dict(page=1,native=True,language='it')})],[dict(id='o',text='non',kind='prose'),dict(id='x',text='extra',kind='prose')],{'g':dict(status='MATCHED',observed_ids=['o'])})
        self.assertEqual(p.text_aggregates(value)[0]['metrics']['CER']['status'],'UNAVAILABLE')
        self.assertEqual(p.native_text_verdict(value),'UNAVAILABLE')

if __name__=='__main__':unittest.main()
