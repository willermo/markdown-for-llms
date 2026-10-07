import json,time,hashlib,sys
from pathlib import Path
H=Path(__file__).parent;RUN=H.parents[2];E=json.loads((H/'entry.json').read_text())
def cost():return E['historical_conservative_entry']+E['initial_reading_charge_seconds']+time.monotonic()-E['monotonic_start']
def write(p,v):p.write_text(json.dumps(v,separators=(',',':')))
def ident(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
write(H/'finalize-failure.json',dict(command=[sys.executable,'-I','-B',str(H/'finalize.py')],exit=1,error='AssertionError charged<8880 after report/checkpoint/manifest writes',workload_started=False,all_real_R_and_mock_results_unchanged=True))
note='\n**Quota prospettica NON PASS:** la stesura e finalizzazione ha oltrepassato il cap8880, intercettato dal gate finale di finalize.py (FAIL conservato). Le due R reali erano concluse a8671.266940s. Nessun nuovo workload oltre cap; i PASS tecnici non attestano PASS del costo. closing-accounting.json misura anche la presente riscrittura con margine1s per ultimo record/print; nessuna sanatoria/reset.\n'
for p in [RUN/'implementation/report-r020.md',RUN/'handovers/implementation-r001.md']:p.write_text(p.read_text()+note)
p=H/'delivery.json';d=json.loads(p.read_text());d['status']='PASS_TARGETED_PROOFS_NONPASS_PROSPECTIVE_TIME';d['report']=ident(RUN/'implementation/report-r020.md');d['checkpoint']=ident(RUN/'handovers/implementation-r001.md');d['closing_script']=ident(Path(__file__));write(p,d)
write(H/'manifest-closing.json',[ident(p) for p in H.iterdir() if p.is_file() and p.name not in ['manifest-closing.json','closing-accounting.json']])
v=cost();write(H/'closing-accounting.json',dict(status='NON_PASS_PROSPECTIVE_TIME',command=[sys.executable,'-I','-B',str(Path(__file__))],script=ident(Path(__file__)),observed_after_all_report_checkpoint_manifest_rewrites=v,closing_record_margin_seconds=1,prospective_charged_upper=v+1,cap=8880,overrun=v+1-8880,historical_lower_bound=8321.418588865316,historical_upper_known=False,historical_budget_PASS=False,new_workload_after_cap=False,wall=time.time(),monotonic=time.monotonic(),no_further_file_writes=True));print('Technical proofs PASS; prospective charge upper',v+1,'overrun',v+1-8880)
