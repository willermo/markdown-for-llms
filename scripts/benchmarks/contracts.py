"""Contratti H01-H16: stub dichiarati, nessuna conversione Pandoc simulata."""
import argparse
from pathlib import Path
import shutil
import sys
import time
import traceback
from data import atomic_new,descriptor,digest,load,write_new
from resources import run
from audit import audit
from benchmark import observe_bundle,score_bundle,run_attempt
from metrics import batch,evaluate_exit
from safety import Unsafe,regular
from campaign import check_file,manifest

EXPECTED={'H01':0,'H02':0,'H03':0,'H04':0,'H05':0,'H06':0,'H07':0,'H08':0,
          'H09':4,'H10-hash':4,'H10-schema':4,'H10-missing':4,'H11':4,'H12-negative':1,'H12-positive':1,'H13':1,'H14':2,'H15':0,'H15-file':4,'H16':130}


def positive(case,corpus,output,environment,installation):
    control=corpus/'controls/K01-full';raw=output/'raw';raw.mkdir()
    source=regular(control,'document.md');(output/'stub-input.md').write_bytes(source)
    (raw/'assets').mkdir();shutil.copyfile(control/'assets/figure-it.png',raw/'assets/figure-it.png')
    if case=='H08':
        document={'document_id':'invalid-input-control','format':'txt','input':'../sentinel','input_sha256':'0'*64}
        result=run_attempt(corpus,document,'txt-literal-v1',{'config_sha256':'test-stub-not-engine'},output/'invalid-attempt','contract-h08','invalid',environment,installation)
        assert result['run_exit']==4 and not result['engine_started'] and result['conversion']=='INVALID_INPUT'
        write_new(output/'case.json',{'case_id':case,'actual':result,'expected':'INVALID_INPUT, no engine started, outside baseline'});return 0
    # This independent trusted fixture script is written before spawn. Only
    # data paths/mode enter argv. It never changes the baseline tool registry.
    stub=output/'stub.py';stub.write_text('''import os,sys,time
from pathlib import Path
payload=Path(sys.argv[1]).read_bytes();mode=sys.argv[3]
if mode=="H02":payload=payload.replace(b"serie 0012",b"serie 012",1)
if mode=="H03":payload=payload.replace(b"E=mc^2",b"E=mc^2\\\\z",1)
if mode=="H05":payload=b""
if mode=="H06":payload+=b"\\n![Unsafe](../sentinel)\\n"
Path(sys.argv[2]).write_bytes(payload)
if mode=="H04":print("Deliberate engine stub error",file=sys.stderr);sys.exit(7)
if mode=="H07":
 child=os.fork()
 if child==0:os.setsid();time.sleep(20);os._exit(0)
 time.sleep(20)
''')
    receipt=run([sys.executable,'-I','-B',str(stub),str(output/'stub-input.md'),str(raw/'document.md'),case],output,output/'engine-stub',environment,.25 if case=='H07' else 5)
    write_new(output/'resources.json',receipt)
    checked=audit(raw);conversion='TIMEOUT' if receipt['timeout'] else 'ERROR' if receipt['adapter_exit'] or receipt['stop_reason'] else 'EMPTY' if checked['reason']=='EMPTY' else 'SUCCESS'
    if conversion in ('TIMEOUT','ERROR') and checked['status']=='COMPLETE':checked.update(status='INCOMPLETE',reason='ENGINE_'+conversion)
    runexit=0 if conversion=='SUCCESS' and checked['status']=='COMPLETE' else 2
    write_new(output/'audit.json',checked);observed=observe_bundle(raw,output/'observations.json')
    measured=score_bundle(control/'annotations.json',output/'observations.json',output/'audit.json',corpus/'assets',output/'metrics.json','control-markdown')
    expected={'H01':(0,5,'SUCCESS','COMPLETE','INDETERMINATE'),'H02':(0,3,'SUCCESS','COMPLETE','FAIL'),'H03':(0,5,'SUCCESS','COMPLETE','INDETERMINATE'),'H04':(2,3,'ERROR','INCOMPLETE','FAIL'),'H05':(2,3,'EMPTY','INCOMPLETE','FAIL'),'H06':(2,3,'SUCCESS','INCOMPLETE','FAIL'),'H07':(2,3,'TIMEOUT','INCOMPLETE','FAIL')}[case]
    actual=(runexit,measured['score_exit'],conversion,checked['status'],measured['quality'])
    write_new(output/'case.json',{'case_id':case,'fixture':'explicit synthetic stub; not native baseline','expected':list(expected),'actual':list(actual),'evaluate_exit':evaluate_exit(runexit,measured['score_exit']),'automatic_gates':measured['automatic_gates'],'batch':batch([{'attempt_id':case,'conversion':conversion,'bundle':checked['status'],'quality':measured['quality']}]),'resources':descriptor(output/'resources.json')})
    assert actual==expected,(actual,expected)
    if case=='H01':assert measured['automatic_gates']=='PASS' and not measured['failures']
    if case=='H02':assert any(f['metric']=='numbers' for f in measured['failures'])
    if case=='H03':assert any(u['metric']=='formula' for u in measured['unavailable']) and not measured['failures']
    if case=='H04':assert receipt['adapter_exit']==7 and (output/'engine-stub/stderr.bin').stat().st_size
    if case=='H06':assert checked['reason']=='UNSAFE_REFERENCE'
    if case=='H07':assert receipt['timeout'] and receipt['children_reaped'] and receipt['stop_reason']=='TIMEOUT'
    return 0


def negative(case,output,environment):
    # Assertions deliberately fail only in the requested fault-injection case;
    # successful inner execution never turns their outer expected0 R green.
    if case=='H09':
        from adapters import validate
        validate(output,{'format':'txt','input':'missing.txt','input_sha256':'0'*64},'txt-literal-v1');return 0
    if case=='H10-hash':
        p=output/'receipt.json';write_new(p,{'state':'original'});before=descriptor(p);p.write_text('{"state":"tampered"}');check_file(before);return 0
    if case=='H10-schema':
        p=output/'manifest.json';write_new(p,{'schema_version':99});manifest(p);return 0
    if case=='H10-missing':load(output/'required-final.json');return 0
    if case=='H11':
        # A programming exception injected into the real scorer call is an
        # internal fault, not an unsupported document: no broad success catch.
        from metrics import score
        score({'blocks':[{'type':'paragraph'}]}, {'nodes':[]}, {'status':'COMPLETE'});return 0
    if case in ('H12-negative','H12-positive'):
        from copy import deepcopy
        corpus=Path(__file__).resolve().parents[2]/'tests/fixtures/conversion-corpus'
        # First collect a genuine discriminating observation/score; then
        # inject a broken comparator decision and assert the frozen contract.
        positive('H02' if case=='H12-negative' else 'H01',corpus,output,environment,None)
        measured=load(output/'metrics.json');faulty=deepcopy(measured)
        if case=='H12-negative':
            faulty.update(failures=[],automatic_gates='PASS',quality='INDETERMINATE',score_exit=5)
            required=3;actual=faulty['score_exit']
        else:
            faulty.update(automatic_gates='FAIL');required='PASS';actual=faulty['automatic_gates']
        write_new(output/'faulty-comparator.json',{'injection':'deliberately broken comparator decision','original_metrics':descriptor(output/'metrics.json'),'faulty':faulty,'expected':required,'actual':actual})
        assert actual==required;return 0
    if case=='H13':
        p=output/'test_deliberate_failure.py';p.write_text('def test_fault_injection():\n    assert 1 == 2\n')
        receipt=run([sys.executable,'-I','-B','-m','pytest',str(p),'-q','--basetemp='+str(output/'pytest-work')],output,output/'pytest',environment,10);write_new(output/'pytest.json',receipt)
        return 1 if receipt['adapter_exit'] else 0
    if case=='H14':
        deadline=time.monotonic()+.1;receipt=run([sys.executable,'-I','-B','-c','import time;time.sleep(20)'],output,output/'harness-process',environment,.1);write_new(output/'resources.json',receipt)
        assert time.monotonic()>deadline and receipt['timeout'] and receipt['children_reaped'];atomic_new(output/'partial.json',{'state':'INCOMPLETE','harness_exit':2});return 2
    if case=='H15':
        # Process0 alone cannot satisfy the external guard. Exceed stdout in
        # the outer R command itself, rather than concealing it in a child log.
        sys.stdout.write('x'*(1048576+8192));sys.stdout.flush();return 0
    if case=='H15-file':
        with (output/'oversize.bin').open('xb') as stream:stream.truncate(33554433)
        return 0
    if case=='H16':
        import signal
        stub='import os,signal,time;os.kill(os.getppid(),signal.SIGINT);time.sleep(20)'
        receipt=run([sys.executable,'-I','-B','-c',stub],output,output/'interrupted-process',environment,5);write_new(output/'resources.json',receipt)
        assert receipt['interrupted'] and receipt['children_reaped'];atomic_new(output/'partial.json',{'state':'INTERRUPTED','conversion_batch':'INTERRUPTED','harness_exit':130});return 130
    raise ValueError('Case not registered')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=EXPECTED,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--environment',type=Path,required=True);p.add_argument('--installation',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);write_new(a.output/'started.json',{'case_id':a.case,'expected_harness_exit':EXPECTED[a.case],'R_expected_exit':0,'fixture':'explicit test-only fault injection, no baseline claim'})
    try:
        env=load(a.environment);env=env.get('environment',env)
        code=positive(a.case,a.corpus,a.output,env,a.installation) if a.case in ('H01','H02','H03','H04','H05','H06','H07','H08') else negative(a.case,a.output,env)
    except AssertionError:traceback.print_exc();code=1
    except KeyboardInterrupt:code=130
    except Exception:traceback.print_exc();code=4
    atomic_new(a.output/'contract-exit.json',{'case_id':a.case,'actual_harness_exit':code,'expected_harness_exit':EXPECTED[a.case],'R_expected_exit':0,'match':code==EXPECTED[a.case]});return code

if __name__=='__main__':raise SystemExit(main())
