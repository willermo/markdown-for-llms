"""Matrice finita r004: raw reali sintetici, post-run mock distinti da R live."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import contextlib
import shutil
import uuid
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parents[3]
H = ROOT/'scripts/diagnostics/run-a003-fase1-engine-comparison'
def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
p = load(H/'text_projection.py'); c = load(H/'confinement.py'); d = load(H/'driver.py')

def annotation(specs, contexts=()):
    segments = []; offset = 0
    for value, op, attrs in specs:
        if not value:
            continue
        end = offset+len(value.encode())
        s = dict(span=[offset,end],raw=value,unit_id='o',operation=op,
                 classification='CONTENT' if op == 'KEEP' else 'SYNTAX',emit=value if op == 'KEEP' else '')
        s.update(attrs); segments.append(s); offset = end
    raw = ''.join(s['raw'] for s in segments).encode()
    return raw, dict(version=p.VERSION,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),segments=segments,contexts=list(contexts),relations={})

def partition(a, variant):
    a = copy.deepcopy(a)
    if variant == 'keep-codepoints':
        segments = []
        for s in a['segments']:
            if s['operation'] != 'KEEP':
                segments.append(s); continue
            off = s['span'][0]
            for ch in s['raw']:
                end = off+len(ch.encode()); segments.append(dict(s,span=[off,end],raw=ch,emit=ch)); off=end
        a['segments'] = segments
    for i, s in enumerate(a['segments']):
        s['unit_id'] = 'renamed' if variant == 'renamed' else 'o' if variant == 'whole' else 'fragment-'+str(i)
    return a

def domain_cases():
    """Quattro operazioni, otto delimitatori e quattro partizioni deterministiche."""
    cases = []
    constructs = [('`','`'),('``','``'),('$','$'),('$$','$$'),('\\(','\\)'),('\\[','\\]'),('~~~\n','\n~~~\n'),('<code>','</code>'),('<pre>','</pre>')]
    for opening, closing in constructs:
        for op in ('T00','T02','T03','T07'):
            if op == 'T00':
                body = [('x','KEEP',{}),('\n','T00',dict(domain='prose',mode='soft-wrap',emit=' ')),('i','KEEP',{})]
            elif op == 'T02':
                body = [('x','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_')),('i','KEEP',{})]
            elif op == 'T03':
                body = [('_','T03',dict(domain='prose',pair_id='pair',side='open')),('x','KEEP',{}),('_','T03',dict(domain='prose',pair_id='pair',side='close'))]
            else:
                body = [('<p>','KEEP',{}),('&amp;','T07',dict(domain='html-text',emit='&')),('</p>','KEEP',{})]
            raw, a = annotation([(opening,'KEEP',{}),*body,(closing,'KEEP',{})])
            a['contexts'] = [dict(kind='prose',span=[0,len(raw)],unit_id='o')]
            if op == 'T07':
                start=len(opening.encode())+3; end=start+5
                a['contexts'].append(dict(kind='html-text',span=[start,end],opening=[start-3,start],closing=[end,end+4],unit_id='o'))
            for variant in ('whole','renamed','separate','keep-codepoints'):
                cases.append(dict(name=repr(opening)+'/'+op+'/'+variant,raw=raw,annotation=partition(a,variant),accepted=False))
    # Invarianti URI con e senza registry e frammentazione fra segmenti/unità.
    for prefix in ('https://e.invalid/','http://e.invalid/','mailto:','www.e.invalid/'):
        for op, body in [('T02',[('x','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_')),('b','KEEP',{})]),
                         ('T03',[('_','T03',dict(domain='prose',pair_id='p',side='open')),('b','KEEP',{}),('_','T03',dict(domain='prose',pair_id='p',side='close'))]),
                         ('T00',[('x ','KEEP',{}),(' ','T00',dict(domain='prose',mode='padding')),('b','KEEP',{})])]:
            raw,a=annotation([(prefix,'KEEP',{}),*body]);a['contexts']=[dict(kind='prose',span=[0,len(raw)],unit_id='o')]
            if prefix != 'www.e.invalid/':
                a['contexts'].append(dict(kind='uri',span=[0,len(raw)],fragmented=True,unit_id='different'))
            for variant in ('whole','renamed','separate','keep-codepoints'):
                cases.append(dict(name=prefix+'/'+op+'/'+variant,raw=raw,annotation=partition(a,variant),accepted=False))
    # Operazione lecita nello stesso raw di una regione protetta; nessun blanket reject.
    for opening,closing in constructs:
        raw,a=annotation([(opening+'x\\_i'+closing+' prosa ','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_')),(' ok','KEEP',{})])
        a['contexts']=[dict(kind='prose',span=[0,len(raw)],unit_id='deliberately-different')]
        for variant in ('whole','renamed','separate','keep-codepoints'):
            cases.append(dict(name='positive-outside/'+repr(opening)+'/'+variant,raw=raw,annotation=partition(a,variant),accepted=True,emission=raw.decode().replace(' prosa \\_', ' prosa _')))
    # Aperture incomplete/non supportate non diventano prosa per dichiarazione.
    for opening in ('~~~','<code ', '<pre ', '$'):
        raw,a=annotation([(opening+'x','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_')),('i','KEEP',{})])
        a['contexts']=[dict(kind='prose',span=[0,len(raw)],unit_id='o')]
        for variant in ('whole','renamed','separate','keep-codepoints'):
            cases.append(dict(name='unsupported-incomplete/'+opening+'/'+variant,raw=raw,annotation=partition(a,variant),accepted=False))
    return cases

def link(dest, label='fonte', after=' ok'):
    raw,a=annotation([('Vedi ','KEEP',{}),('[','T04',dict(role='delimiter',relation='r',evidence='finite test')),
                      (label,'KEEP',{}),('](','T04',dict(role='delimiter',relation='r',evidence='finite test')),
                      (dest,'T04',dict(role='destination',relation='r',evidence='finite test')),
                      (')','T04',dict(role='delimiter',relation='r',evidence='finite test')),(after,'KEEP',{})])
    segs=[s for s in a['segments'] if s['operation']=='T04'];span=[segs[0]['span'][1],segs[1]['span'][0]]
    a['relations']={'r':dict(kind='link',channel='references',segments=[s['span'] for s in segs],label_span=span,label=label,target=dest)}
    return raw,a

def result(raw,a):
    try:
        v=p.project(raw,a)
        return dict(accepted=True,emission=''.join(v['units'].values()),projection=v)
    except ValueError as exc:
        return dict(accepted=False,error=str(exc))

class RawContract(unittest.TestCase):
    def test_partition_domain_matrix(self):
        for case in domain_cases():
            with self.subTest(name=case['name']):
                actual=result(case['raw'],case['annotation'])
                self.assertEqual(actual['accepted'],case['accepted'],actual)
                if case['accepted']:self.assertEqual(actual['emission'],case['emission'])
    def test_contradictory_overlap_and_utf8(self):
        raw,a=annotation([('prosa ','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_'))]);a['contexts']=[dict(kind='prose',span=[0,len(raw)],unit_id='o')]
        a['segments'][1]['channel']='literal'
        self.assertFalse(result(raw,a)['accepted'])
        raw,a=annotation([('é','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_'))]);a['contexts']=[dict(kind='prose',span=[1,len(raw)],unit_id='o')]
        self.assertFalse(result(raw,a)['accepted'])
    def test_intraword_flanking_unpaired(self):
        for prefix,body,suffix in [('a','b','c'),('snake','case','name'),('é','β','文'),('',' b',''),('','b ',''),('','','')]:
            raw,a=annotation([(prefix,'KEEP',{}),('_','T03',dict(domain='prose',pair_id='p',side='open')),(body,'KEEP',{}),('_','T03',dict(domain='prose',pair_id='p',side='close')),(suffix,'KEEP',{})])
            for variant in ('whole','separate'):
                with self.subTest(raw=raw,variant=variant):self.assertFalse(result(raw,partition(a,variant))['accepted'])
    def test_T00_internal_padding_and_breaks(self):
        for text,op,mode,emit,ok in [('A  B','  ','padding','',False),(' A',' ','padding','',True),('A ',' ','padding','',True),('A\nB','\n','soft-wrap',' ',True),('A\r\nB','\r\n','crlf','\n',True),('A  \nB','\n','soft-wrap',' ',False)]:
            i=text.index(op);raw,a=annotation([(text[:i],'KEEP',{}),(op,'T00',dict(domain='prose',mode=mode,emit=emit)),(text[i+len(op):],'KEEP',{})])
            for variant in ('whole','separate','keep-codepoints'):
                with self.subTest(text=text,variant=variant):self.assertEqual(result(raw,partition(a,variant))['accepted'],ok)
        raw,a=annotation([('A\nB','KEEP',{})]);self.assertEqual(result(raw,a)['emission'],'A\nB')
    def test_N6_actual_bytes_and_KEEP(self):
        # Due backslash prima di $, esattamente i byte del probe Claude.
        raw,a=annotation([('a \\\\$x','KEEP',{}),('\\_','T02',dict(domain='prose',emit='_')),('i$','KEEP',{})]);a['contexts']=[dict(kind='prose',span=[0,len(raw)],unit_id='o')]
        self.assertFalse(result(raw,a)['accepted'])
        raw,a=annotation([(raw.decode(),'KEEP',{})]);self.assertEqual(result(raw,a)['emission'],raw.decode())
    def test_destination_finite_matrix(self):
        destinations=[('https://e.invalid/a_b?x=0012',True),('a(b)c',True),('<https://e.invalid/a>',True),
                      ('https://e.invalid/a) non (b',False),('x\nnon si',False),('x y',False),('x\tq',False),('x\x00q',False),('x\x7fq',False),('x\u00a0q',False),('a(b',False),('a)b',False),('x "title"',False),('<x\ny>',False),('<x<y>',False),('<x> y',False),('<x\u200bq>',False)]
        for dest,ok in destinations:
            for label in ('fonte','non','源'):
                raw,a=link(dest,label)
                for variant in ('whole','separate'):
                    with self.subTest(dest=dest,label=label,variant=variant):
                        actual=result(raw,partition(a,variant));self.assertEqual(actual['accepted'],ok,actual)
                        if ok:self.assertEqual(actual['emission'],'Vedi '+label+' ok')
    def test_syntax_inside_literal_and_external_fence(self):
        cases=[]
        for specs in [ [('`','KEEP',{}),('# ','T01',dict(wrapper='heading',evidence='false prose')),('non`','KEEP',{})],
                       [('`','KEEP',{}),('<p>','T06',dict(evidence='tag')),('non','KEEP',{}),('</p>','T06',dict(evidence='tag')),('`','KEEP',{})],
                       [('`A','KEEP',{}),('\n\n','T08',dict(kind='blank-layout',evidence='false layout')),('B`','KEEP',{})] ]:
            cases.append(annotation(specs))
        raw,a=link('https://e.invalid/a')
        for s in a['segments']:s['span']=[v+1 for v in s['span']]
        for rel in a['relations'].values():
            rel['segments']=[[v+1 for v in span] for span in rel['segments']];rel['label_span']=[v+1 for v in rel['label_span']]
        a['segments']=[dict(span=[0,1],raw='`',emit='`',operation='KEEP',classification='CONTENT',unit_id='o'),*a['segments'],dict(span=[len(raw)+1,len(raw)+2],raw='`',emit='`',operation='KEEP',classification='CONTENT',unit_id='o')]
        raw=b'`'+raw+b'`';a.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());cases.append((raw,a))
        for raw,a in cases:
            for variant in ('whole','renamed','separate','keep-codepoints'):
                with self.subTest(raw=raw,variant=variant):self.assertFalse(result(raw,partition(a,variant))['accepted'])
        raw,a=annotation([('````text\n','T01',dict(wrapper='fence',evidence='external fence')),('a_b $x_i$\n','KEEP',dict(domain='literal',channel='literal')),('````\n','T01',dict(wrapper='fence',evidence='external fence'))])
        self.assertEqual(result(raw,a)['emission'],'a_b $x_i$\n')

@contextlib.contextmanager
def attempt():
    parent=c.RUN/'work'/c.DELIVERY/'attempts';parent.mkdir(parents=True,exist_ok=True)
    leaf=parent/('unit-'+uuid.uuid4().hex);leaf.mkdir()
    try:yield str(leaf)
    finally:
        shutil.rmtree(leaf)
        shutil.rmtree(c.RUN/'evidence'/c.DELIVERY/'attempts'/leaf.name,ignore_errors=True)
def manifest(scratch):
    evidence=c.RUN/'evidence'/c.DELIVERY/'attempts'/scratch.name;evidence.mkdir(parents=True,exist_ok=True)
    config=scratch/'consumer.json';config.write_text('{}')
    return dict(staging={'consumer.json':c.descriptor(config)},schema=1,phase='conversion',scratch=str(scratch),readonly=['/usr'],readonly_leaf=[str(config)],files=[],argv=['/usr/bin/true'],
                python='/usr/bin/python3',consumer=str(H/'consumer.py'),consumer_config=str(scratch/'consumer.json'),
                environment={k:str(scratch) for k in ('HOME','TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME')},
                process_output=str(evidence/'consumer-process'),result=str(evidence/'consumer-result.json'))

class ScratchContract(unittest.TestCase):
    def test_RW_and_RO_roots(self):
        with attempt() as td:
            scratch=Path(td);m=manifest(scratch);self.assertEqual(c.validate(m),scratch)
            roots=[ROOT,ROOT/'tests/fixtures/conversion-corpus/gold',Path('/home/davide'),ROOT/'temp',c.RUN,scratch.parent,scratch.parent.parent,Path('/tmp')]
            for root in roots:
                with self.subTest(root=str(root)):
                    bad=copy.deepcopy(m);bad['scratch']=str(root);bad['environment']={k:str(root) for k in bad['environment']}
                    with self.assertRaises(ValueError):c.argv(bad)
                    bad=copy.deepcopy(m);bad['readonly'].append(str(root))
                    with self.assertRaises(ValueError):c.argv(bad)
    def test_env_links_traversal_sibling_and_input_overlay(self):
        with attempt() as td,attempt() as sibling:
            root=Path(td);m=manifest(root);(root/'source').mkdir();(root/'source/input.txt').write_text('immutable');m['readonly_leaf'].append(str(root/'source/input.txt'));m['staging']['source/input.txt']=c.descriptor(root/'source/input.txt')
            args=c.argv(m);self.assertLess(args.index('--bind'),args.index('--ro-bind'));self.assertIn(str(root/'source/input.txt'),args)
            (root/'link').symlink_to(root.parent)
            for key in m['environment']:
                for value in [str(root.parent),str(root/'link'/'cache'),str(root)+'/../x',str(root)+'/./cache',str(root)+'//cache']:
                    with self.subTest(key=key,value=value):
                        bad=copy.deepcopy(m);bad['environment'][key]=value
                        with self.assertRaises(ValueError):c.validate(bad)
            bad=copy.deepcopy(m);bad['scratch']=sibling;bad['environment']={k:sibling for k in bad['environment']};bad['consumer_config']=str(Path(sibling)/'consumer.json')
            with self.assertRaisesRegex(ValueError,'relation'):c.validate(bad)
            bad=copy.deepcopy(m);bad['scratch']=str(root/'link')
            with self.assertRaises(ValueError):c.validate(bad)
            bad=copy.deepcopy(m);bad['scratch']=str(root)+'/../'+root.name
            with self.assertRaises(ValueError):c.validate(bad)
            bad=copy.deepcopy(m);bad['environment']['XDG_CONFIG_DIRS']=str(root)+':/usr/share'
            with self.assertRaises(ValueError):c.validate(bad)
    def test_setup_exact_destination(self):
        root=c.RUN/'work/implementation-r002/setup-ml-r001'
        m=dict(setup_writable=[str(root/'marker')],schema=1,phase='setup',scratch=str(root),readonly=['/usr'],files=[],argv=['/usr/bin/true'],environment={k:str(root) for k in ('HOME','TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME')});self.assertEqual(c.validate(m),root)
        for path in (root.parent,root/'marker',ROOT,Path('/tmp')):
            bad=copy.deepcopy(m);bad['scratch']=str(path);bad['environment']={k:str(path) for k in bad['environment']}
            with self.assertRaises(ValueError):c.validate(bad)

class PostRunContract(unittest.TestCase):
    def fixture(self, root):
        scratch=root/'scratch';scratch.mkdir();e=root/'evidence';(e/'R').mkdir(parents=True)
        (e/'R/admission.json').write_text('{"status":"PASS"}');(e/'R/receipt.json').write_text('{"status":"PASS"}')
        (e/'consumer-result.json').write_text('{"adapter_exit":0,"stop_reason":null}')
        canonical=root/'canonical';canonical.write_bytes(b'input');leaf=scratch/'input';leaf.write_bytes(b'input')
        (scratch/'document.md').write_bytes(b'raw');outside=root/'outside';outside.write_bytes(b'UNREAD')
        (scratch/'outside-link').symlink_to(outside);info=(scratch/'outside-link').lstat()
        snapshot=root/'snapshot';snapshot.write_text('{}')
        return dict(evidence=e,scratch=scratch,safety=SimpleNamespace(MAX_FILE=1024*1024),attempt_id='mock-r004',case='synthetic',engine='txt-literal-v1',
                    wrapper=dict(adapter_exit=0,children_reaped=True),snapshot=snapshot,sentinel=dict(identity=dict(dev=info.st_dev,ino=info.st_ino),target=str(outside)),
                    canonical=c.descriptor(canonical),canonical_path=canonical,input_leaf=leaf,payload=b'input')
    def test_sentinel_absent_directory_regular_and_normal(self):
        for mode in ('normal','absent','directory','regular','changed-link'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);kw=self.fixture(root);sentinel=kw['scratch']/'outside-link'
                if mode!='normal':
                    sentinel.unlink()
                    if mode=='directory':sentinel.mkdir();(sentinel/'retained').write_text('partial')
                    elif mode=='regular':sentinel.write_text('replaced')
                    elif mode=='changed-link':sentinel.symlink_to(root/'canonical')
                value=d.finalize_attempt(**kw)
                self.assertEqual(value['operational_status'],'SUCCESS' if mode=='normal' else 'ERROR')
                self.assertEqual(json.loads((kw['evidence']/'final.json').read_text()),value)
                self.assertEqual((root/'outside').read_bytes(),b'UNREAD');self.assertTrue(value['integrity_unchanged'])
                if mode=='directory':self.assertEqual((sentinel/'retained').read_text(),'partial')
                self.assertTrue((kw['evidence']/'consumer-result.json').exists())
    def test_copy_failure_and_replacement(self):
        for mode in ('copy-write','same-bytes-new-inode','safe-read'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                root=Path(td);kw=self.fixture(root);original=d.artifact_inventory
                def altered(*args):
                    rows=original(*args)
                    if mode=='copy-write':(kw['evidence']/'raw/document.md').mkdir(parents=True)
                    elif mode=='same-bytes-new-inode':
                        replacement=kw['scratch']/'replacement';replacement.write_bytes(b'raw');os.replace(replacement,kw['scratch']/'document.md')
                    elif mode=='safe-read':(kw['scratch']/'document.md').unlink()
                    return rows
                with mock.patch.object(d,'artifact_inventory',side_effect=altered):value=d.finalize_attempt(**kw)
                self.assertEqual(value['operational_status'],'ERROR');self.assertTrue(value['output_read_errors']);self.assertEqual(value['raw']['bytes'],3)
                self.assertTrue(value['artifacts']);self.assertTrue((kw['evidence']/'final.json').exists());self.assertEqual((root/'outside').read_bytes(),b'UNREAD')
    def test_final_unwritable_host_channel(self):
        with tempfile.TemporaryDirectory() as td:
            kw=self.fixture(Path(td));(kw['evidence']/'final.json').mkdir()
            with mock.patch.object(d.sys,'stderr') as stream,self.assertRaises(OSError):d.finalize_attempt(**kw)
            self.assertIn('final-record-write',str(stream.write.call_args_list));self.assertTrue((kw['evidence']/'consumer-result.json').exists())

if __name__=='__main__':unittest.main()
