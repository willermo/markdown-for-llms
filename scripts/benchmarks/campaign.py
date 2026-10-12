"""Harness: manifest anteriore, receipt obbligatorie, exit0 R fisso."""
import argparse
from pathlib import Path
import sys
import time
import traceback
from data import atomic_new,canonical,descriptor,digest,load,write_new
from inventory import verify
from metrics import batch,evaluate_exit
from profile import config
from resources import run
from safety import no_symlink_root,regular,relative

PATHS=(('direct-txt','C01-txt-it','txt-literal-v1'),('direct-docx','C02-docx-it','pandoc-direct-v1'),('direct-epub','C03-epub-en','pandoc-direct-v1'),('direct-rtf','C04-rtf-it','pandoc-direct-v1'),('direct-tex','C05-tex-en','pandoc-direct-v1'),('legacy-docx','C02-docx-it','pandoc-legacy-v1'),('legacy-epub','C03-epub-en','pandoc-legacy-v1'),('legacy-rtf','C04-rtf-it','pandoc-legacy-v1'))
LEGITIMATE=(0,2,3,5)


def check_file(row):
    path=Path(row['path']);regular(path.parent,path.name)
    if descriptor(path)!=row:raise ValueError('STALE descriptor: '+str(path))
    return path


def manifest(path):
    value=load(path)
    if value.get('schema_version')!=1 or value.get('protocol')!='conversion-fidelity-v1' or value.get('kind')!='campaign':raise ValueError('Campaign schema')
    if digest({k:v for k,v in value.items() if k!='manifest_sha256'})!=value['manifest_sha256']:raise ValueError('Campaign digest')
    for row in value['protected_inputs']:check_file(row)
    verify(load(check_file(value['external_inputs'])))
    registry=[(g['group_id'],g['document_id'],g['engine_id']) for g in value['groups']]
    if value.get('scope','full') not in ('full','dependent-rerun'):raise ValueError('Campaign scope')
    expected=list(PATHS) if value.get('scope','full')=='full' else [row for row in PATHS if row[0] in {g[0] for g in registry}]
    if not registry or registry!=expected or any(len(g['attempts'])!=4 for g in value['groups']):raise ValueError('Campaign registry / four attempts per path')
    if value['expected']['records_required']!=4*len(registry):raise ValueError('Campaign expected records')
    ids=[a for g in value['groups'] for a in g['attempts']]
    if len(set(ids))!=4*len(registry):raise ValueError('Duplicate attempts')
    for ident in ids:
        relative(ident)
        if '/' in ident or len(ident)>80:raise ValueError('Attempt syntax')
    return value


def freeze(corpus,stage,output,experiment,selected=None,verification=None):
    no_symlink_root(output.parent);output.mkdir(exist_ok=False)
    relative(experiment)
    if '/' in experiment:raise ValueError('Experiment syntax')
    corpus=corpus.absolute();stage=stage.absolute();documents=load(corpus/'manifest.json')['documents']
    protected=[descriptor(p) for p in sorted(corpus.rglob('*')) if p.is_file()]
    write_new(output/'environment.json',load(stage/'target.json')['environment']);groups=[]
    if selected is not None and (not selected or len(set(selected))!=len(selected) or set(selected)-{p[0] for p in PATHS}):raise ValueError('Unknown/duplicate dependent path')
    paths=[p for p in PATHS if selected is None or p[0] in selected]
    for name,ident,engine in paths:
        document=next(d for d in documents if d['document_id']==ident)
        docpath=output/(name+'-source.json');cfgpath=output/(name+'-config.json')
        write_new(docpath,{k:document[k] for k in ('document_id','format','input','input_sha256')});write_new(cfgpath,config(document['format'],engine))
        groups.append({'group_id':name,'document_id':ident,'engine_id':engine,'document':descriptor(docpath),'config':descriptor(cfgpath),'gold':descriptor(corpus/document['gold']),'attempts':[experiment+'-'+name+'-r'+str(i) for i in range(1,5)]})
    if verification:
        for g in groups:
            receipt=Path(verification)/(g['document_id']+'.json');v=load(receipt)
            if v.get('status')!='VERIFIED' or v.get('gold_sha256')!=g['gold']['sha256'] or v.get('canonical_sha256')!=load(g['document']['path'])['input_sha256']:raise ValueError('Reference qualification STALE or not VERIFIED')
            g['verification']=descriptor(receipt);protected.append(g['verification'])
    protected.extend(descriptor(p) for p in sorted(output.glob('*.json')))
    value={'schema_version':1,'protocol':'conversion-fidelity-v1','kind':'campaign','experiment_id':experiment,'scope':'full' if selected is None else 'dependent-rerun','created_utc':__import__('resources').utc(),'corpus':str(corpus),'assets':str(corpus/'assets'),'python':str(stage/'dev/bin/python'),'installation':descriptor(stage/'I.json'),'external_inputs':descriptor(stage/'benchmark-inputs.json'),'environment':descriptor(output/'environment.json'),'protected_inputs':protected+[descriptor(stage/'I.json')],'groups':groups,'expected':{'harness_exit':0,'R_expected_exit':0,'evaluate_exits':list(LEGITIMATE),'records_required':4*len(paths)},'deadlines_s':{'engine':60,'harness':400,'R_proposed':600},'workspace':'fresh_process_fresh_workspace for every attempt','os_cache':'uncontrolled','quality_required_human':'reference qualification frozen when supplied; formula output judgments remain unassigned','reference_qualification':'VERIFIED frozen receipt' if verification else 'PROVISIONAL'}
    value['manifest_sha256']=digest(value);write_new(output/'manifest.json',value);return value


def check_record(path,g,ident,m,actual_exit):
    row=load(path)
    required=('schema_version','experiment_id','document_id','attempt_id','engine_id','source_sha256','config_sha256','protocol','wheel','installation','reader','gold_sha256','observations','conversion','bundle','quality','run_exit','score_exit','evaluate_exit','run','metrics','audit','automatic_gates')
    if any(k not in row for k in required) or row['schema_version']!=1:raise ValueError('Final record schema')
    source=load(check_file(g['document']));cfg=load(check_file(g['config']))
    expected={'experiment_id':m['experiment_id'],'document_id':g['document_id'],'attempt_id':ident,'engine_id':g['engine_id'],'source_sha256':source['input_sha256'],'config_sha256':cfg['config_sha256'],'gold_sha256':g['gold']['sha256'],'protocol':m['protocol'],'installation':m['installation']}
    if any(row[k]!=v for k,v in expected.items()):raise ValueError('Final record identity mismatch')
    for key in ('run','metrics','audit','observations','reader'):check_file(row[key])
    runrecord=load(row['run']['path']);metrics=load(row['metrics']['path']);observed=load(row['observations']['path'])
    if digest(observed)!=metrics['observations_sha256']:raise ValueError('Observation/scorer digest')
    if row['wheel']!=load(m['installation']['path'])['payload']['installed']['canonical_archive']:raise ValueError('Wheel identity')
    if row['run_exit'] not in (0,2) or row['score_exit'] not in (0,3,5):raise ValueError('Internal/invalid input is not motor loss')
    if row['evaluate_exit']!=actual_exit or evaluate_exit(row['run_exit'],row['score_exit'])!=actual_exit or actual_exit not in LEGITIMATE:raise ValueError('Exit precedence')
    if any(row[k]!=runrecord[k] for k in ('conversion','bundle','run_exit')) or any(row[k]!=metrics[k] for k in ('quality','score_exit','automatic_gates')):raise ValueError('State receipt mismatch')
    if row['run_exit']==0 and (row['conversion'],row['bundle'])!=('SUCCESS','COMPLETE'):raise ValueError('False success')
    if row['run_exit']==2 and (row['conversion'],row['bundle'])==('SUCCESS','COMPLETE'):raise ValueError('Contradictory operational error')
    if metrics['score_exit']!={'PASS':0,'FAIL':3,'INDETERMINATE':5}[metrics['quality']]:raise ValueError('Quality/score contract')
    resource=runrecord['adapter']['resources']
    if not resource['children_reaped']:raise RuntimeError('Uncollected motor descendants')
    if row['conversion']=='TIMEOUT' and not resource['timeout']:raise ValueError('False timeout')
    if row['conversion']=='ERROR' and not (resource['adapter_exit'] or resource['stop_reason']):raise ValueError('False error')
    return row


def group(path,name,output):
    m=manifest(path)
    if sys.executable!=m['python']:raise ValueError('Python differs from installation')
    g=next(g for g in m['groups'] if g['group_id']==name)
    no_symlink_root(output.parent);output.mkdir(exist_ok=False)
    started=time.monotonic_ns();deadline=time.monotonic()+m['deadlines_s']['harness']
    write_new(output/'started.json',{'schema_version':1,'manifest':descriptor(path),'group_id':name,'expected_exit':0,'state':'STARTED'})
    records=[];receipts=[]
    try:
        for ident in g['attempts']:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Harness deadline before attempt')
            attempt=output/ident
            argv=[sys.executable,'-I','-B',str(Path(__file__).with_name('launch.py')),'benchmark','evaluate','--corpus',m['corpus'],'--document',g['document']['path'],'--engine',g['engine_id'],'--config',g['config']['path'],'--output',str(attempt),'--experiment-id',m['experiment_id'],'--attempt-id',ident,'--environment',m['environment']['path'],'--installation',m['installation']['path'],'--gold',g['gold']['path'],'--assets',m['assets']]
            if g.get('verification'):argv+=['--verification',str(check_file(g['verification']))]
            receipt=run(argv,output,output/(ident+'-evaluate-process'),load(m['environment']['path']),min(remaining,270))
            write_new(output/(ident+'-evaluate.json'),receipt);receipts.append(descriptor(output/(ident+'-evaluate.json')))
            if receipt.get('interrupted'):raise KeyboardInterrupt
            if receipt['timeout'] or receipt['stop_reason'] or time.monotonic()>deadline:raise TimeoutError('Harness monitor/deadline stopped attempt')
            if receipt['adapter_exit'] not in LEGITIMATE:raise ValueError('Non-motor failure: evaluate exit '+str(receipt['adapter_exit']))
            row=check_record(attempt/'final.json',g,ident,m,receipt['adapter_exit']);records.append(row)
            with (output/'ledger.jsonl').open('ab') as stream:stream.write(canonical({'attempt_id':ident,'final':descriptor(attempt/'final.json'),'evaluate_process':receipts[-1]})+b'\n')
        manifest(path)
        result={'schema_version':1,'group_id':name,'manifest':descriptor(path),'execution_complete':True,'harness_exit':0,'records':[descriptor(output/a/'final.json') for a in g['attempts']],'evaluate_resources':receipts,'elapsed_s':(time.monotonic_ns()-started)/1e9,**batch(records)}
        atomic_new(output/'group.json',result);return 0
    except (KeyboardInterrupt,TimeoutError) as exc:
        code=130 if isinstance(exc,KeyboardInterrupt) else 2
        atomic_new(output/'partial.json',{'schema_version':1,'state':'INTERRUPTED' if code==130 else 'INCOMPLETE','conversion_batch':'INTERRUPTED' if code==130 else 'PARTIAL' if records else 'FAILED','harness_exit':code,'reason':str(exc),'records_completed':len(records),'evaluate_resources':receipts});return code
    except Exception:
        traceback.print_exc();atomic_new(output/'partial.json',{'schema_version':1,'state':'INVALID','harness_exit':4,'records_completed':len(records),'evaluate_resources':receipts});return 4


def summarize(path,groups,output):
    m=manifest(path)
    if len(groups)!=len(m['groups']):raise ValueError('All declared group receipts required')
    allrecords=[];a10=[]
    for g,p in zip(m['groups'],groups):
        value=load(p)
        if value['group_id']!=g['group_id'] or value['manifest']!=descriptor(path) or value['harness_exit']!=0 or value['execution_complete'] is not True or len(value['records'])!=4:raise ValueError('Incomplete group')
        local=[]
        for ident,r in zip(g['attempts'],value['records']):
            check_file(r);row=check_record(Path(r['path']),g,ident,m,load(r['path'])['evaluate_exit']);local.append(row)
        allrecords.extend(local)
        if g['document_id'] in ('C02-docx-it','C03-epub-en'):
            eligible=[]
            for row in local:
                metrics=load(row['metrics']['path']);expected={b['asset'] for b in load(g['gold']['path'])['blocks'] if b['type']=='figure'}
                actual={a['expected_asset'] for a in metrics['asset_verifications'] if a.get('content_verified') and a.get('uri')}
                eligible.append(row['conversion']=='SUCCESS' and row['bundle']=='COMPLETE' and bool(expected) and expected.issubset(actual))
            a10.append({'group_id':g['group_id'],'four_complete_with_expected_referenced_figures':all(eligible)})
    result={'schema_version':1,'manifest':descriptor(path),'execution_complete':len(allrecords)==m['expected']['records_required'],'harness_exit':0,'groups':[descriptor(p) for p in groups],**batch(allrecords),'A10':{'satisfied':any(r['four_complete_with_expected_referenced_figures'] for r in a10),'paths':a10},'quality_human':m.get('reference_qualification','PROVISIONAL')+' / UNAVAILABLE formula output judgments'}
    atomic_new(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    f=s.add_parser('freeze')
    for n in ('corpus','stage','output'):f.add_argument('--'+n,type=Path,required=True)
    f.add_argument('--experiment-id',required=True);f.add_argument('--groups',nargs='+');f.add_argument('--verification',type=Path)
    r=s.add_parser('run-group');r.add_argument('--manifest',type=Path,required=True);r.add_argument('--group',required=True);r.add_argument('--output',type=Path,required=True)
    a=s.add_parser('summarize');a.add_argument('--manifest',type=Path,required=True);a.add_argument('--groups',type=Path,nargs='+',required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.command=='freeze':freeze(args.corpus,args.stage,args.output,args.experiment_id,args.groups,args.verification);return 0
        if args.command=='run-group':return group(args.manifest,args.group,args.output)
        print(summarize(args.manifest,args.groups,args.output)['conversion_batch']);return 0
    except KeyboardInterrupt:return 130
    except Exception:traceback.print_exc();return 4

if __name__=='__main__':raise SystemExit(main())
