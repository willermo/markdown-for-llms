"""Negativi sintetici/mock r003; prove live conservate separatamente nella run."""
import errno,fcntl,hashlib,importlib.util,json,os
from pathlib import Path
import tempfile,unittest
from unittest import mock
ROOT=Path(__file__).absolute().parents[3];H=ROOT/'scripts/diagnostics/run-a003-fase1-engine-comparison'
def load(path):
 spec=importlib.util.spec_from_file_location(path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
p=load(H/'text_projection.py');c=load(H/'confinement.py');d=load(H/'driver.py');consumer=load(H/'consumer.py');resources=load(ROOT/'scripts/benchmarks/resources.py')
def annotation(raw,specs,**extra):
 segments=[];offset=0
 for text,op,attrs in specs:
  end=offset+len(text.encode());emit=text if op=='KEEP' else ''
  seg=dict(raw=text,span=[offset,end],unit_id='o',operation=op,classification='CONTENT' if op=='KEEP' else 'SYNTAX',emit=emit);seg.update(attrs);segments.append(seg);offset=end
 return dict(version=p.VERSION,sha256=hashlib.sha256(raw.encode()).hexdigest(),bytes=len(raw.encode()),segments=segments,**extra)
class ProjectionFixes(unittest.TestCase):
 def test_duplicate_in_and_across_rows(self):
  obs=[dict(id='o',kind='prose',text='A')]
  for gold,align in [([dict(id='g',kind='prose',text='AA')],{'g':dict(status='MATCHED',observed_ids=['o','o'])}),([dict(id=i,kind='prose',text='A') for i in ['g','h']],{i:dict(status='MATCHED',observed_ids=['o']) for i in ['g','h']})]:
   with self.assertRaisesRegex(ValueError,'duplicated'):p.assess(gold,obs,align)
 def test_distinct_fragments(self):
  r=p.assess([dict(id='g',kind='prose',text='AB')],[dict(id='a',kind='prose',text='A'),dict(id='b',kind='prose',text='B')],{'g':dict(status='MATCHED',observed_ids=['a','b'])});self.assertEqual(r['units'][0]['measurement']['CER']['distance'],0)
 def test_invented_relation_cannot_erase_negation(self):
  a=annotation('non si',[('non ','T04',dict(relation='invented',evidence='declared',role='delimiter')),('si','KEEP',{})])
  with self.assertRaises(ValueError):p.project(b'non si',a)
  a['relations']={'invented':dict(kind='anchor',channel='references',segments=[[0,4]])}
  with self.assertRaises(ValueError):p.project(b'non si',a)
 def test_crossed_and_nested_pairs(self):
  for crossed in [False,True]:
   raw='**a _b'+('** c_' if crossed else '_ c**')
   a=annotation(raw,[('**','T03',dict(pair_id='x',side='open',domain='prose')),('a ','KEEP',{}),('_','T03',dict(pair_id='y',side='open',domain='prose')),('b','KEEP',{}),('**' if crossed else '_','T03',dict(pair_id='x' if crossed else 'y',side='close',domain='prose')),(' c','KEEP',{}),('_' if crossed else '**','T03',dict(pair_id='y' if crossed else 'x',side='close',domain='prose'))])
   if crossed:
    with self.assertRaisesRegex(ValueError,'Crossed'):p.project(raw.encode(),a)
   else:self.assertEqual(p.project(raw.encode(),a)['units']['o'],'a b c')
 def test_HTML_entity_context_and_invalid(self):
  for entity,emit in [('&amp;','&'),('&#x31;','1'),('&bogus;','?'),('&#xD800;','?')]:
   raw='<p>'+entity+'</p>';a=annotation(raw,[('<p>','T06',dict(evidence='tag')), (entity,'T07',dict(emit=emit,domain='html-text')),('</p>','T06',dict(evidence='tag'))],contexts=[dict(kind='html-text',unit_id='o',span=[3,3+len(entity)],opening=[0,3],closing=[3+len(entity),len(raw)])])
   if emit=='?':
    with self.assertRaises(ValueError):p.project(raw.encode(),a)
   else:self.assertEqual(p.project(raw.encode(),a)['units']['o'],emit)
 def test_URI_escape_and_entity_cannot_change_domain(self):
  for value,op,emit,domain in [('\\_','T02','_','prose'),('&amp;','T07','&','html-text')]:
   raw='https://example.invalid/a'+value+'b';start=len('https://example.invalid/a');a=annotation(raw,[(raw[:start],'KEEP',{}),(value,op,dict(emit=emit,domain=domain)),('b','KEEP',{})],contexts=[dict(kind='prose',unit_id='o',span=[0,len(raw)])])
   with self.assertRaises(ValueError):p.project(raw.encode(),a)
   a['segments'][1].update(operation='KEEP',classification='CONTENT',emit=value);self.assertEqual(p.project(raw.encode(),a)['units']['o'],raw)
 def test_non_HTML_entity_rejected_and_numbers_code_keep(self):
  raw='0012 $x_i$ `a_b` &amp;';a=annotation(raw,[(raw,'KEEP',{})]);self.assertEqual(p.project(raw.encode(),a)['units']['o'],raw)
  bad=annotation('&amp;',[('&amp;','T07',dict(domain='html-text',emit='&'))])
  with self.assertRaises(ValueError):p.project(b'&amp;',bad)
 def test_finite_context_and_declaration_cannot_be_forged(self):
  for raw,prefix,suffix in [('$x\\_i$','$x','i$'),('`a\\_b`','`a','b`')]:
   start=len(prefix);a=annotation(raw,[(prefix,'KEEP',{}),('\\_','T02',dict(domain='prose',emit='_')),(suffix,'KEEP',{})],contexts=[dict(kind='prose',span=[0,len(raw)],unit_id='o')])
   with self.assertRaises(ValueError):p.project(raw.encode(),a)
  raw='{#actual}';a=annotation(raw,[(raw,'T04',dict(relation='anchor',role='declaration',evidence='synthetic'))],relations={'anchor':dict(kind='anchor',channel='references',segments=[[0,len(raw)]],declarations=['{#invented}'])})
  with self.assertRaises(ValueError):p.project(raw.encode(),a)
  a['relations']['anchor']['declarations']=[raw];self.assertEqual(p.project(raw.encode(),a)['units']['o'],'')

class OutputFixes(unittest.TestCase):
 def test_safe_output_regular_and_negatives(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);outside=root/'outside';outside.write_bytes(b'UNREAD_SYNTHETIC');sub=root/'scratch';sub.mkdir();(sub/'good').write_text('{"status":"PASS"}')
   self.assertEqual(c.safe_json(sub,'good')['status'],'PASS')
   (sub/'link').symlink_to(outside);(sub/'ancestor').symlink_to(root);os.link(outside,sub/'hard');os.mkfifo(sub/'fifo');(sub/'large').write_bytes(b'x'*33)
   for name,limit in [('link',1024),('ancestor/outside',1024),('hard',1024),('fifo',1024),('large',32),('absent',1024)]:
    with self.assertRaises((ValueError,OSError)):c.safe_bytes(sub,name,limit)
   (sub/'invalid').write_text('{')
   with self.assertRaises(ValueError):c.safe_json(sub,'invalid')
   # A replacement during read cannot retain the original object's identity.
   original_read=os.read;swapped=[False]
   def replace(fd,n):
    chunk=original_read(fd,n)
    if chunk and not swapped[0]:
     swapped[0]=True;(sub/'replacement').write_text('other');os.replace(sub/'replacement',sub/'good')
    return chunk
   with mock.patch.object(c.os,'read',side_effect=replace),self.assertRaises(ValueError):c.safe_bytes(sub,'good')
 def test_seal_authentic_missing_malformed_and_replacement(self):
  fd=c.memfd_create()
  try:
   with self.assertRaises(ValueError):c.sealed_admission(fd)
   data=json.dumps(dict(status='PASS',phase='consumer-before-exec',probes=[dict(passed=True)])).encode();os.write(fd,data);fcntl.fcntl(fd,c.F_ADD_SEALS,c.SEALS)
   self.assertEqual(c.sealed_admission(fd)[0],data)
   with self.assertRaises(OSError):os.pwrite(fd,b'false PASS',0)
  finally:os.close(fd)
  for raw in [b'',b'{',b'{}',json.dumps(dict(status='PASS',phase='consumer-before-exec',probes=[dict(passed=False)])).encode()]:
   fd=c.memfd_create()
   try:
    os.write(fd,raw);fcntl.fcntl(fd,c.F_ADD_SEALS,c.SEALS)
    with self.assertRaises(ValueError):c.sealed_admission(fd)
   finally:os.close(fd)
class GuardFixes(unittest.TestCase):
 def test_regular_same_bytes_replacement_rejected_by_inode(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);path=root/'receipt';path.write_bytes(b'authentic');info=path.stat();identity={'dev':info.st_dev,'ino':info.st_ino};self.assertEqual(c.safe_bytes(root,'receipt',expected_identity=identity),b'authentic');(root/'replacement').write_bytes(b'authentic');os.replace(root/'replacement',path)
   with self.assertRaises(ValueError):c.safe_bytes(root,'receipt',expected_identity=identity)

 def test_bind_allowlist(self):
  run=ROOT/'temp/run-a003-fase1-engine-comparison'
  for path in [ROOT,ROOT/'tests/fixtures/conversion-corpus/gold',ROOT.parent,Path('/home/davide'),ROOT/'temp',run/'work/implementation-r003/attempts',Path('/home')]:self.assertFalse(c.allowed_bind(path))
  for path in [Path('/usr'),Path('/etc/fonts'),ROOT/'.venv-python/cpython-3.12.13-linux-x86_64-gnu',run/'work/implementation-r002/setup-ml-r001/marker',run/'work/implementation-r002/docling-models',run/'work/implementation-r001/marker-models-r001']:self.assertTrue(c.allowed_bind(path))
 def test_errno_contracts(self):
  for num in [errno.ETXTBSY,errno.EIO,errno.EEXIST,errno.ENOENT]:self.assertFalse(consumer.denied_errno(OSError(num,'synthetic')))
  self.assertTrue(consumer.denied_errno(OSError(errno.EROFS,'synthetic')));self.assertTrue(consumer.denied_errno(OSError(errno.ENOENT,'synthetic'),absence=True))
 def test_RW_create_is_detected(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'synthetic';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd);self.assertTrue(path.exists());path.unlink()
 def test_wrapper_timeout_boundaries(self):
  for child,want in [(120,330),(450,660),(690,900)]:self.assertEqual(d.wrapper_budget(child),want);self.assertGreater(want,child)
  for child in [691,900,0,-1,True,'120']:
   with self.assertRaises(ValueError):d.wrapper_budget(child)
 def test_no_SUCCESS_on_R_wrapper_integrity_fail(self):
  good=dict(adapter_exit=0,children_reaped=True)
  for wrapper,r,integrity in [(good,{},True),(good,dict(status='FAIL'),True),(good,dict(status='PASS'),False),(dict(adapter_exit=2,children_reaped=True),dict(status='PASS'),True),(dict(adapter_exit=0,children_reaped=False),dict(status='PASS'),True)]:self.assertEqual(d.final_status('SUCCESS',wrapper,r,integrity),'ERROR')
 def test_RSS_invalid_before_launch_and_default(self):
  import inspect
  self.assertEqual(inspect.signature(resources.run).parameters['rss_limit_bytes'].default,6*1024**3)
  for value in [0,-1,True,1.0,'1',None]:
   with self.assertRaises(ValueError):resources.run(['/usr/bin/true'],'/tmp',Path('/tmp/never-created'),{},rss_limit_bytes=value)

class RunnerFailClosed(unittest.TestCase):
 def test_preflight_FAIL_without_boundary(self):
  from types import SimpleNamespace
  runner=load(ROOT/'scripts/diagnostics/run-a001-fase0-uv/run_offline.py')
  parent=c.RUN/'work'/c.DELIVERY/'attempts';parent.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent.parent) as td:
   root=Path(td);scratch=parent/('unit-'+__import__('uuid').uuid4().hex);scratch.mkdir();self.addCleanup(__import__('shutil').rmtree,scratch);out=root/'R';out.mkdir();py='/usr/bin/python3';argv=['/usr/bin/true'];target=dict(python=py,prefix='/usr',base_prefix='/usr',python_version='3.12.13',dependencies=[],startup={},daemons=[])
   cfg=dict(runtime={k:target[k] for k in ('python','prefix','base_prefix','python_version','dependencies','startup')},daemon_paths=[],argv=argv);(scratch/'consumer.json').write_text(json.dumps(cfg));env={k:str(scratch) for k in ['HOME','TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME']}
   manifest=dict(schema=1,phase='conversion',scratch=str(scratch),readonly=['/usr'],files=[],argv=argv,environment=env,python=py,consumer=str(H/'consumer.py'),consumer_config=str(scratch/'consumer.json'))
   evidence=c.RUN/'evidence'/c.DELIVERY/'attempts'/scratch.name;evidence.mkdir(parents=True,exist_ok=True);self.addCleanup(__import__('shutil').rmtree,evidence);manifest.update(process_output=str(evidence/'consumer-process'),result=str(evidence/'consumer-result.json'),staging={'consumer.json':c.descriptor(scratch/'consumer.json')},readonly_leaf=[str(scratch/'consumer.json')])
   mf=root/'manifest.json';mf.write_text(json.dumps(manifest));pref=root/'preflight.json';pref.write_text(json.dumps(dict(schema=1,commands=[[py,'-c','preflight stub']])));args=SimpleNamespace(bundle=None,inside=False,command=[],confinement=mf,preflight_config=pref,repo=ROOT,python=py,cwd=scratch,output=out)
   helper=SimpleNamespace(prerequisites=lambda t:[dict(matches=True)])
   with mock.patch.object(runner,'clean_env',return_value={}),mock.patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=2,stdout='mock preflight FAIL',stderr='')) as launch:
    self.assertEqual(runner.confined_command(args,target,helper,'/tmp/synthetic'),2);self.assertEqual(launch.call_count,1)
   self.assertEqual(json.loads((out/'inside.json').read_text())['status'],'FAIL');self.assertFalse((out/'admission.json').exists())
 def test_missing_preflight_without_boundary(self):
  from types import SimpleNamespace
  runner=load(ROOT/'scripts/diagnostics/run-a001-fase0-uv/run_offline.py')
  args=SimpleNamespace(bundle=None,inside=False,command=[],confinement=None,preflight_config=None)
  with mock.patch.object(runner.subprocess,'run') as launch,self.assertRaises(ValueError):runner.confined_command(args,{},None,'/tmp/s')
  launch.assert_not_called()
 def test_consumer_probe_FAIL_without_engine_exec(self):
  import sys
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);positive=root/'positive';positive.write_bytes(b'synthetic');cfg=dict(argv=['/usr/bin/true'],scratch=str(root),readable=[str(positive)],denied_read=[str(root/'absent')],denied_write=[str(root/'RW-sentinel')],readonly_existing=[],network_helper=str(ROOT/'scripts/diagnostics/run-a001-fase0-uv/check_runner.py'),host_netns=os.readlink('/proc/self/ns/net'),host_pidns=os.readlink('/proc/self/ns/pid'),probe_receipt=str(root/'probes.json'));path=root/'consumer.json';path.write_text(json.dumps(cfg))
   from types import SimpleNamespace
   helper=SimpleNamespace(network_state=lambda:dict(no_external_routes_or_addresses=False,netns='mock-net'),connect_only=lambda *a:dict(connected=False,errno=errno.ENOENT))
   spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None))
   class Pair:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def send(self,data):return len(data)
    def recv(self,n):return b'x'
   with mock.patch.object(sys,'argv',['consumer',str(path)]),mock.patch.object(consumer.importlib.util,'spec_from_file_location',return_value=spec),mock.patch.object(consumer.importlib.util,'module_from_spec',return_value=helper),mock.patch.object(consumer.socket,'socketpair',return_value=(Pair(),Pair())),mock.patch.object(consumer.os,'execve') as execute:
    self.assertEqual(consumer.main(),3);execute.assert_not_called()
   value=json.loads((root/'probes.json').read_text());self.assertEqual(value['status'],'FAIL');self.assertFalse(next(p for p in value['probes'] if p['name'].startswith('denied-write:'))['passed']);self.assertFalse((root/'RW-sentinel').exists())

if __name__=='__main__':unittest.main()
