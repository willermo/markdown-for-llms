"""Misure Linux dei soli processi propri: wait4 e RSS albero campionato."""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import hashlib
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def _process(pid):
    root = Path("/proc") / str(pid)
    raw = (root / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    # fields start at proc stat field 3: starttime22, rss24.
    return {"pid": pid, "starttime": int(fields[19]), "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            "state": fields[0], "ppid": int(fields[1]), "pgrp": int(fields[2]), "session": int(fields[3]),
            "children": [int(x) for t in (root / "task").iterdir() for x in (t / "children").read_text().split()]}


def _signal_owned(item, signum):
    """PID fd prevents signalling a recycled PID after a /proc sample."""
    # The pinned CPython build lacks the optional os/signal wrappers. Linux
    # libc exposes these same kernel operations without a package or fallback.
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    libc.pidfd_open.argtypes=[ctypes.c_int,ctypes.c_uint]
    libc.pidfd_open.restype=ctypes.c_int
    fd=libc.pidfd_open(item['pid'],0)
    if fd<0:
        error=ctypes.get_errno()
        if error==3:return
        raise OSError(error,'pidfd_open')
    try:
        try:
            current=_process(item['pid'])
        except (OSError,ValueError):
            return
        if current['starttime']!=item['starttime']:
            return
        libc.pidfd_send_signal.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint]
        libc.pidfd_send_signal.restype=ctypes.c_int
        if libc.pidfd_send_signal(fd,signum,None,0)<0:
            error=ctypes.get_errno()
            if error!=3:raise OSError(error,'pidfd_send_signal')
    finally:
        os.close(fd)


def _run(argv, cwd, output, environment, timeout, stream_cap, cleanup_executables, interrupted,
         stdout_name='stdout.bin',stderr_name='stderr.bin',expected_netns=None):
    """Guard the owned process tree until all writers exit and are reaped.

    Invocations are serial within a caller. Preexisting children are excluded;
    only the new command's observed descendants and subreaper adoptions enter
    this execution. Signals use pidfds plus the sampled start identity.
    """
    if not argv or not Path(argv[0]).is_absolute() or not 0 < timeout <= 900:
        raise ValueError('argv/deadline invalido')
    output=Path(output);output.mkdir(exist_ok=False)
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'prctl subreaper')
    start,started=time.monotonic_ns(),utc()
    samples,observed,missed,reason=[],{},set(),None
    baseline_children=set(_process(os.getpid())['children'])
    reaped=[];helper_cleanup=[];status=usage=None;leader_end=None;stop_started=None
    with (output/stdout_name).open('xb') as stdout,(output/stderr_name).open('xb') as stderr:
        proc=subprocess.Popen(argv,cwd=cwd,env=environment,shell=False,close_fds=True,start_new_session=True,stdout=stdout,stderr=stderr)
        while True:
            roots=[p for p in _process(os.getpid())['children'] if p not in baseline_children]
            pending=list(roots);current={};rss=0
            for old in observed.values():
                try:
                    item=_process(old['pid'])
                    if item['starttime']==old['starttime']:pending.append(item['pid'])
                except (OSError,ValueError):pass
            while pending:
                pid=pending.pop()
                if pid in current:continue
                try:
                    item=_process(pid);pending.extend(item.pop('children'))
                    try:item['netns']=os.readlink('/proc/'+str(pid)+'/ns/net')
                    except OSError:item['netns']=None
                    try:item['cmdline']=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                    except OSError:item['cmdline']=''
                    current[pid]=item;rss+=item['rss_bytes']
                    key=str(pid)+':'+str(item['starttime'])
                    if key not in observed:observed[key]=dict(item)
                    else:
                        observed[key]['rss_bytes']=max(observed[key]['rss_bytes'],item['rss_bytes'])
                        if item['netns']:observed[key]['netns']=item['netns']
                except (OSError,ValueError):missed.add(pid)
            now=time.monotonic_ns();samples.append({'elapsed_s':(now-start)/1e9,'tree_rss_bytes':rss,'processes':len(current)})
            if status is None:
                pid,rawstatus,rusage=os.wait4(proc.pid,os.WNOHANG)
                if pid:
                    status,usage=rawstatus,rusage;leader_end=now
                    proc.returncode=os.waitstatus_to_exitcode(status)
            for pid,item in current.items():
                if pid==proc.pid or item['ppid']!=os.getpid():continue
                try:
                    collected,childstatus,childuse=os.wait4(pid,os.WNOHANG)
                    if collected:reaped.append({'pid':pid,'exit':os.waitstatus_to_exitcode(childstatus),'user_s':childuse.ru_utime,'system_s':childuse.ru_stime,'maxrss_bytes':childuse.ru_maxrss*1024})
                except ChildProcessError:pass
            # Evaluate every guard also after leader exit, before quiescence.
            if interrupted[0]:reason=reason or 'INTERRUPTED'
            elif os.fstat(stdout.fileno()).st_size>stream_cap or os.fstat(stderr.fileno()).st_size>stream_cap:reason=reason or 'STREAM_LIMIT'
            elif (now-start)/1e9>=timeout:reason=reason or 'TIMEOUT'
            elif rss>6*1024**3:reason=reason or 'RSS_STOP'
            elif expected_netns and any(i['netns'] and i['netns']!=expected_netns for i in current.values()):reason=reason or 'NAMESPACE_MISMATCH'
            if status is not None and leader_end is not None and (now-leader_end)/1e9>=0.25 and not reason:
                for item in current.values():
                    if item['state']=='Z':continue
                    try:executable=os.readlink('/proc/'+str(item['pid'])+'/exe')
                    except OSError:continue
                    accepted=next((r for r in cleanup_executables if r['path']==executable and hashlib.sha256(Path(executable).read_bytes()).hexdigest()==r['sha256']),None)
                    if accepted:
                        if not any(r['pid']==item['pid'] for r in helper_cleanup):helper_cleanup.append({'pid':item['pid'],'executable':accepted,'method':'owned descendant killed by pidfd and wait4 reaped after primary exit'})
                        _signal_owned(item,signal.SIGKILL)
                    # Other owned children may finish normally up to the
                    # command deadline. Leader exit is not a shorter timeout.
            if reason:
                stop_started=stop_started or now
                signum=signal.SIGKILL if (now-stop_started)/1e9>=0.25 else signal.SIGTERM
                for item in current.values():
                    if item['state']!='Z':_signal_owned(item,signum)
                if (now-stop_started)/1e9>5:raise RuntimeError('Processi propri non raccolti: '+repr(current))
            # A just-reaped zombie still appears in this sample: rescan before
            # declaring quiescence. No late writer survives a green receipt.
            if status is not None and not current and not roots:break
            time.sleep(0.01)
        if os.fstat(stdout.fileno()).st_size>stream_cap or os.fstat(stderr.fileno()).st_size>stream_cap:reason=reason or 'STREAM_LIMIT'
    if any((output/n).stat().st_size>stream_cap for n in (stdout_name,stderr_name)):reason=reason or 'STREAM_LIMIT'
    return {'argv':argv,'cwd':str(cwd),'started_utc':started,'ended_utc':utc(),'monotonic_start_ns':start,
            'pid':proc.pid,'conversion_elapsed_s':(time.monotonic_ns()-start)/1e9,
            'adapter_exit':proc.returncode,'signal':-proc.returncode if proc.returncode<0 else None,
            'timeout':reason=='TIMEOUT','interrupted':interrupted[0],'stop_reason':reason,'children_reaped':True,
            'wait4':{'user_s':usage.ru_utime,'system_s':usage.ru_stime,'individual_or_reaped_descendant_maxrss_bytes':usage.ru_maxrss*1024},
            'orphan_children_wait4':reaped,'sampled_tree_peak_rss_bytes':max(s['tree_rss_bytes'] for s in samples),
            'known_helper_cleanup':helper_cleanup,'rss_samples':samples,'observed_processes':list(observed.values()),'missed_pids':sorted(missed),
            'sample_interval_s':0.01,'limitations':['serial invocation within caller','shared pages may be counted repeatedly','short processes and between-sample peaks may be missed','wait4 includes descendants only when reaped','not a hard aggregate memory cap'],
            'stdout':stdout_name,'stderr':stderr_name}


def _private_cleanup():
    """Exception cleanup in a private supervisor which has no foreign child."""
    deadline=time.monotonic()+5
    while True:
        roots=_process(os.getpid())['children']
        if not roots:return
        pending=list(roots);seen=set()
        while pending:
            pid=pending.pop()
            if pid in seen:continue
            seen.add(pid)
            try:
                item=_process(pid);pending.extend(item.pop('children'))
                if item['state']!='Z':_signal_owned(item,signal.SIGKILL)
            except (OSError,ValueError):pass
        for pid in roots:
            try:os.wait4(pid,os.WNOHANG)
            except ChildProcessError:pass
        if time.monotonic()>deadline:raise RuntimeError('Private supervisor cleanup incomplete')
        time.sleep(.01)


def _worker(config_path):
    import json
    cfg=json.loads(Path(config_path).read_text());interrupted=[False]
    def interrupt(signum,frame):interrupted[0]=True
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    try:
        result=_run(cfg['argv'],cfg['cwd'],cfg['output'],cfg['environment'],cfg['timeout'],cfg['stream_cap'],cfg['cleanup_executables'],interrupted,cfg['stdout_name'],cfg['stderr_name'],cfg['expected_netns'])
        with Path(cfg['receipt']).open('x') as f:json.dump(result,f);f.write('\n')
    finally:_private_cleanup()


def run(argv, cwd, output, environment, timeout=60, stream_cap=1048576, cleanup_executables=(), *, stdout_name="stdout.bin", stderr_name="stderr.bin", expected_netns=None):
    """One private supervisor per command: foreign caller children untouched.

    The stdlib supervisor is in the same interpreter, closed environment and
    namespace as the caller. It alone adopts, signals and reaps its workload.
    SIGINT is forwarded by pidfd and consumed until cleanup has completed.
    """
    import json
    import sys
    output=Path(output).absolute();prefix=output.with_name(output.name+'-monitor')
    if output.exists():raise FileExistsError(output)
    config_path=prefix.with_name(prefix.name+'-input.json');receipt_path=prefix.with_name(prefix.name+'-receipt.json')
    cfg={'argv':list(map(str,argv)),'cwd':str(cwd),'output':str(output),'environment':environment,'timeout':timeout,'stream_cap':stream_cap,'cleanup_executables':cleanup_executables,'stdout_name':stdout_name,'stderr_name':stderr_name,'expected_netns':expected_netns,'receipt':str(receipt_path)}
    with config_path.open('x') as f:json.dump(cfg,f);f.write('\n')
    supervisor_argv=[sys.executable,'-I','-B',str(Path(__file__).absolute()),'--private-supervisor',str(config_path)]
    previous=signal.getsignal(signal.SIGINT);interrupted=[False];worker_item=None
    def interrupt(signum,frame):
        interrupted[0]=True
        if worker_item is not None:_signal_owned(worker_item,signal.SIGINT)
    signal.signal(signal.SIGINT,interrupt)
    try:
        with prefix.with_name(prefix.name+'-stdout.bin').open('xb') as stdout,prefix.with_name(prefix.name+'-stderr.bin').open('xb') as stderr:
            proc=subprocess.Popen(supervisor_argv,env=environment,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True)
            try:worker_item=_process(proc.pid)
            except OSError:pass
            if interrupted[0] and worker_item:_signal_owned(worker_item,signal.SIGINT)
            while True:
                pid,status,usage=os.wait4(proc.pid,os.WNOHANG)
                if pid:break
                time.sleep(.01)
            proc.returncode=os.waitstatus_to_exitcode(status)
        if proc.returncode or not receipt_path.is_file():raise RuntimeError('Private supervisor failed: '+str(proc.returncode)+'; see '+str(prefix)+'-stderr.bin')
        result=json.loads(receipt_path.read_text())
        if interrupted[0]:result.update(interrupted=True,stop_reason=result.get('stop_reason') or 'INTERRUPTED')
        result['supervisor']={'argv':supervisor_argv,'pid':proc.pid,'config':str(config_path),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'receipt':str(receipt_path),'exit':proc.returncode,'foreign_caller_children_signal_or_reap':False}
        result['supervisor_wait4']={'user_s':usage.ru_utime,'system_s':usage.ru_stime,'individual_or_reaped_descendant_maxrss_bytes':usage.ru_maxrss*1024}
        result['limitations']=[x for x in result['limitations'] if x!='serial invocation within caller']+['private per-command supervisor; no concurrent-child ownership inference in caller']
        return result
    finally:signal.signal(signal.SIGINT,previous)


if __name__=='__main__':
    import sys
    if not sys.flags.isolated or not sys.dont_write_bytecode or len(sys.argv)!=3 or sys.argv[1]!='--private-supervisor':raise SystemExit('Private supervisor requires -I -B')
    _worker(sys.argv[2])
