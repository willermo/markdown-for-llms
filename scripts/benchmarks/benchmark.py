"""CLI del benchmark locale: run/observe/score separati, nessun dominio fase2."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
import traceback

from adapters import validate,convert
from pandoc_guard import check as pandoc_check
from audit import audit
from data import atomic_new,canonical,descriptor,digest,digest_bytes,load,write_new,counts
from metrics import score as measure,evaluate_exit,batch
from observations import observe
from profile import config
from safety import Unsafe,regular,no_symlink_root,relative


def run_attempt(corpus,document,engine,cfg,output,experiment,attempt,environment,installation):
    started=time.monotonic_ns();output=output.absolute()
    if any(k in document for k in ('gold','blocks','gold_sha256')):raise ValueError('Run source descriptor must be gold-blind')
    for name in (experiment,attempt,document['document_id']):
        relative(name)
        if len(name)>80 or '/' in name:raise ValueError('Identity syntax')
    no_symlink_root(output.parent);output.mkdir(mode=0o700,exist_ok=False)
    identity={'experiment_id':experiment,'document_id':document['document_id'],'attempt_id':attempt,'source_sha256':document['input_sha256'],'engine_id':engine,'config_sha256':cfg['config_sha256'],'protocol':'conversion-fidelity-v1','wheel':load(installation)['payload']['installed']['canonical_archive'],'installation':descriptor(installation),'reader':descriptor(Path(__file__).with_name('observations.py'))}
    write_new(output/'started.json',{'schema_version':1,**identity,'state':'STARTED'})
    try:
        validate(corpus,document,engine)
        if engine!='txt-literal-v1':pandoc_check(environment,cfg['pandoc'],cfg['pandoc_version_required'])
    except (ValueError,UnicodeError,OSError) as exc:
        result={'schema_version':1,**identity,'conversion':'INVALID_INPUT','bundle':'UNAVAILABLE','run_exit':4,'engine_started':False,'error':str(exc),'raw':None,'audit':{'status':'UNAVAILABLE','reason':'INVALID_INPUT'},'end_to_end_s':(time.monotonic_ns()-started)/1e9};atomic_new(output/'run.json',result);return result
    try:
        actual=convert(corpus,document,engine,cfg,output/'raw',environment,installation)
    except KeyboardInterrupt:
        result={'schema_version':1,**identity,'conversion':'INTERRUPTED','bundle':'UNAVAILABLE','run_exit':130,'engine_started':True,'raw':'raw','audit':{'status':'UNAVAILABLE','reason':'INTERRUPTED'}};atomic_new(output/'run.json',result);return result
    resource=actual['resources'];conversion='INTERRUPTED' if resource.get('interrupted') else 'TIMEOUT' if resource['timeout'] else 'ERROR' if resource['adapter_exit'] or resource['stop_reason'] else 'SUCCESS'
    audit_start=time.monotonic_ns();checked=audit(output/'raw');audit_elapsed=(time.monotonic_ns()-audit_start)/1e9
    if conversion=='SUCCESS' and checked['reason']=='EMPTY':conversion='EMPTY'
    if conversion not in ('SUCCESS','EMPTY') and checked['status']=='COMPLETE':
        checked.update(status='INCOMPLETE',reason='ENGINE_'+conversion)
    exit=130 if conversion=='INTERRUPTED' else 0 if conversion=='SUCCESS' and checked['status']=='COMPLETE' else 2
    if regular(corpus,document['input'])!=regular(output/'raw/source','main.tex' if document['format']=='tex' else 'input.'+document['format']):raise ValueError('Source changed across execution')
    write_new(output/'audit.json',checked)
    result={'schema_version':1,**identity,'conversion':conversion,'bundle':checked['status'],'run_exit':exit,'engine_started':True,'raw':'raw','adapter':actual,'audit':checked,'audit_elapsed_s':audit_elapsed,'end_to_end_s':(time.monotonic_ns()-started)/1e9}
    atomic_new(output/'run.json',result);return result


def observe_bundle(root,output):
    try:content=regular(root,'document.md')
    except FileNotFoundError:content=b''
    try:result=observe(content)
    except (Unsafe,UnicodeError) as exc:
        text=content.decode('utf-8',errors='replace');result={'schema_version':1,'reader':'observations-v1','raw_sha256':digest_bytes(content),'raw_bytes':len(content),'nodes':[{'observation_id':'o0001','type':'unknown','text':text,'raw':text,'offset':{'byte_start':0,'byte_end':len(content),'char_start':0,'char_end':len(text),'line_start':1,'line_end':text.count('\n')+1},'references':[],'anchors':[],'formulas':[],'reason':'INVALID_ENCODING' if isinstance(exc,UnicodeError) else 'UNSAFE_REFERENCE'}],'unknown':1,'unsafe':str(exc),'raw_hex':content.hex() if isinstance(exc,UnicodeError) else None}
    write_new(output,result);return result


def score_bundle(goldpath,observedpath,auditpath,assets,output,format,verification_path=None):
    gold=load(goldpath);observed=load(observedpath);checked=deepcopy(load(auditpath))
    if checked.get('raw_sha256') is not None and checked['raw_sha256']!=observed['raw_sha256']:raise ValueError('Audit/observations identity STALE')
    # Gold reference bytes are read only in this score process. Raw observations
    # and audit remain byte-identical when an alternative gold is supplied.
    expected={}
    for b in gold['blocks']:
        leaves=[c for row in b.get('cell_contents',[]) for c in row]+([b['caption_content']] if b.get('caption_content') else [])
        names=([b['asset']] if b.get('asset') else [])+[r['asset'] for c in leaves for r in c.get('references',[]) if r.get('kind')=='image' and r.get('asset')]
        for name in names:
            if name not in expected:
                payload=regular(assets,name);expected[name]=(digest_bytes(payload),payload)
    for image in checked.get('assets',[]):
        image['content_verified']=False
        for name,(sha,payload) in expected.items():
            method=None
            if image.get('sha256')==sha:method='SHA256 exact original PNG bytes'
            elif image.get('path','').endswith('.png'):
                from png_codec import pixel_digest
                rawroot=Path(auditpath).parent/'raw';actual=regular(rawroot,image['path'])
                if pixel_digest(actual)==pixel_digest(payload):method='lossless normalized RGBA pixel SHA256'
            if method:image.update(content_verified=True,expected_asset=name,verification_method=method,expected_sha256=sha);break
    verification=load(verification_path) if verification_path else None
    if verification and (verification.get('gold_sha256')!=descriptor(goldpath)['sha256'] or verification.get('canonical_sha256')!=gold['canonical_sha256']):verification={'status':'STALE'}
    result=measure(gold,observed,checked,format,verification)
    result['gold_sha256']=descriptor(goldpath)['sha256'];result['audit_sha256']=descriptor(auditpath)['sha256'];result['asset_verifications']=checked.get('assets',[])
    write_new(output,result);return result


def execute(argv,cwd,output,environment):
    # The role subprocess argv is constant trusted code, not document content.
    from resources import run
    receipt=run(argv,cwd,output,environment,90);write_new(output.with_suffix('.json'),receipt)
    if receipt.get('interrupted'):raise KeyboardInterrupt
    return receipt['adapter_exit']


def evaluate(a):
    launcher=Path(__file__).with_name('launch.py');base=[sys.executable,'-I','-B',str(launcher),'benchmark'];a.output.parent.mkdir(parents=True,exist_ok=True)
    role=a.output.parent/(a.output.name+'-roles');role.mkdir(exist_ok=False)
    argv=base+['run','--corpus',str(a.corpus),'--document',str(a.document),'--engine',a.engine,'--config',str(a.config),'--output',str(a.output),'--experiment-id',a.experiment_id,'--attempt-id',a.attempt_id,'--environment',str(a.environment),'--installation',str(a.installation)]
    supplied=load(a.environment);environment=supplied.get('environment',supplied)
    run_exit=execute(argv,a.output.parent,role/'run',environment)
    if run_exit not in (0,2,4,130):raise ValueError('Run internal failure, see role traceback')
    if not (a.output/'run.json').exists():raise ValueError('Run receipt missing')
    if run_exit in (4,130):return run_exit
    observe_exit=execute(base+['observe','--bundle',str(a.output/'raw'),'--output',str(a.output/'observations.json')],a.output.parent,role/'observe',environment)
    if observe_exit!=0:raise ValueError('Reader internal failure')
    scoreargv=base+['score','--gold',str(a.gold),'--observations',str(a.output/'observations.json'),'--audit',str(a.output/'audit.json'),'--assets',str(a.assets),'--format',load(a.document)['format'],'--output',str(a.output/'metrics.json')]
    if a.verification:scoreargv+=['--verification',str(a.verification)]
    score_exit=execute(scoreargv,a.output.parent,role/'score',environment)
    if score_exit not in (0,3,5):raise ValueError('Scorer internal failure')
    runrecord=load(a.output/'run.json');metrics=load(a.output/'metrics.json');obs=descriptor(a.output/'observations.json')
    if digest(load(a.output/'observations.json'))!=metrics['observations_sha256']:raise ValueError('Observation digest changed')
    result={'schema_version':1,**{k:runrecord[k] for k in ('experiment_id','document_id','attempt_id','engine_id','source_sha256','config_sha256','protocol','wheel','installation','reader')},'gold_sha256':descriptor(a.gold)['sha256'],'observations':obs,'conversion':runrecord['conversion'],'bundle':runrecord['bundle'],'quality':metrics['quality'],'run_exit':run_exit,'score_exit':score_exit,'evaluate_exit':evaluate_exit(run_exit,score_exit),'run':descriptor(a.output/'run.json'),'metrics':descriptor(a.output/'metrics.json'),'audit':descriptor(a.output/'audit.json'),'automatic_gates':metrics['automatic_gates'],'required_output_human_judgments':'UNAVAILABLE/PENDING; not assigned'}
    atomic_new(a.output/'final.json',result);return result['evaluate_exit']


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    i=s.add_parser('inventory');i.add_argument('--corpus',type=Path,required=True)
    c=s.add_parser('profile');c.add_argument('--format',required=True);c.add_argument('--engine',required=True);c.add_argument('--output',type=Path,required=True)
    for command in ('run','evaluate'):
        r=s.add_parser(command)
        for name in ('corpus','document','config','output','environment','installation'):r.add_argument('--'+name,type=Path,required=True)
        for name in ('engine','experiment-id','attempt-id'):r.add_argument('--'+name,required=True)
        if command=='evaluate':
            for name in ('gold','assets'):r.add_argument('--'+name,type=Path,required=True)
            r.add_argument('--verification',type=Path)
    o=s.add_parser('observe');o.add_argument('--bundle',type=Path,required=True);o.add_argument('--output',type=Path,required=True)
    t=s.add_parser('score')
    for name in ('gold','observations','audit','assets','output'):t.add_argument('--'+name,type=Path,required=True)
    t.add_argument('--format',required=True);t.add_argument('--verification',type=Path)
    u=s.add_parser('summarize');u.add_argument('--records',type=Path,nargs='+',required=True);u.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.command=='inventory':print(canonical({'manifest':descriptor(a.corpus/'manifest.json'),'documents':[{'id':d['document_id'],'source':d['input_sha256'],'gold':d['gold_sha256']} for d in load(a.corpus/'manifest.json')['documents']]}).decode());return 0
        if a.command=='profile':write_new(a.output,config(a.format,a.engine));return 0
        if a.command=='run':
            env=load(a.environment);r=run_attempt(a.corpus.absolute(),load(a.document),a.engine,load(a.config),a.output,a.experiment_id,a.attempt_id,env.get('environment',env),a.installation);print(r['conversion'],r['bundle']);return r['run_exit']
        if a.command=='observe':observe_bundle(a.bundle,a.output);return 0
        if a.command=='score':r=score_bundle(a.gold,a.observations,a.audit,a.assets,a.output,a.format,a.verification);print(r['quality'],r['automatic_gates']);return r['score_exit']
        if a.command=='evaluate':return evaluate(a)
        if a.command=='summarize':write_new(a.output,batch([load(path) for path in a.records]));return 0
    except Exception as exc:
        # Contract/internal failure is exit4, never a recognized motor-loss.
        traceback.print_exc();return 4

if __name__=='__main__':raise SystemExit(main())
