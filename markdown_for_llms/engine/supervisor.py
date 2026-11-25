"""The supervisor of a child container (MFL051): PID 1 of every child, imposed by the launcher.

The launcher starts every child with `/bin/bash -c SUPERVISOR mdf-supervise MAX LEASE TICK CTL STAT CGROUP --`
as its entrypoint (`entrypoint`); the command of the request follows as arguments and cannot replace it. The
script is the same for the image of the application and for the toolchain image, which has bash and no Python.
It starts the command and, at every tick, with builtins only:

- `CTL/stop` exists: exit 202 (the caller stops the child without Docker);
- the counter in `CTL/lease` has not changed for LEASE ticks: exit 200 (the caller is stuck or dead);
- more than MAX ticks: exit 201 (the absolute term of the request);
- every fifth tick, a sample of its own cgroup, as the container sees it: `memory.peak`, `usage_usec` of
  `cpu.stat`, `oom_kill` of `memory.events`.

When PID 1 of a PID namespace exits the kernel kills every process of the namespace: the container ends without
the launcher, the caller or Docker. Lease and term are counted in ticks, never read from a clock, so a suspended
laptop does not expire them. `CTL` is a read-only mount of its own, outside every writable mount of the child.

The files of `STAT` (`stdout`, `stderr`, `metrics`, `exit`) are created here, exclusively, before the command
exists, and written only through the descriptors opened then: a path that the command removes or replaces (with
a FIFO too) is never opened again, so it cannot block the supervisor. A file already there is a refusal (203):
the caller hands over an empty directory. The command inherits none of the supervisor's descriptors; its stdin
is `CTL/stdin` when the caller wrote one (a regular file, not a link), else `/dev/null`. Without a `STAT`
directory (`-`, the `manifest` kind) the two streams go to the log of the container and the metrics nowhere.

What it is not: a security boundary against code running in the child, which has the same user and namespaces.
A signal from the command either never reaches it (the kernel drops, for the init of a PID namespace, the signals
it has no handler for: `KILL`, `STOP`, and `QUIT`, which bash ignores) or ends it at once: `TERM`, `INT`, `HUP`,
`USR1`, `USR2`, `PIPE` and `ALRM`, which bash handles as PID 1 whether a script traps them or not, have one trap
here and end it with 143, an earlier exit; a writer on its pipe makes the ticks faster, an earlier exit again;
`stat/` is written by the command too, so metrics and `stat/exit` are hints for the caller, never proof: the end
of a child is what Docker says through the launcher. When the supervisor itself is defeated, the limits of the
kernel and the launcher's own watch remain.

The order of the script is part of it. The variables (`child` too, empty until there is a command) and the two
functions come first; the trap is set as soon as `metrics` and `exit` are open, before the two streams, the stdin
and the command. So a signal that arrives before the command exists finds `finish` defined, kills nothing, and
ends the supervisor with 143 and `stat/exit` written; `finish` disarms the trap when it starts, so a second
signal does not run it again. Before the trap is set the script has run for an instant only, and a signal then
is bash's own to handle: the container ends, without `stat/exit`.

The command gets the environment of the container: `PWD` and `OLDPWD` of the shell are not exported. Three names
come on top of the image's and the request's: `HOSTNAME`, set by Docker, and `SHLVL` and `_`, which bash sets for
a command it starts in the background.

A line of `metrics` is `mdf-metrics/1 <peak bytes> <cpu usec> <oom kills> <tick> end`, one `printf` per sample,
appended; the reader takes the last complete line (`last_metrics`). A value the cgroup does not give is `-`.
"""
import re

BASH = "/bin/bash"
ARG0 = "mdf-supervise"
TICK_SECONDS = 0.1
CTL, STAT, CGROUP = "/ctl", "/stat", "/sys/fs/cgroup"
EXIT_SIGNAL, EXIT_LEASE, EXIT_TERM, EXIT_STOP, EXIT_STAT = 143, 200, 201, 202, 203
STREAMS = ("stdout", "stderr", "metrics", "exit")

SUPERVISOR = r'''set -u
max=$1 lease=$2 tick=$3 ctl=$4 stat=$5 cg=$6
shift 7
n=0 seen=0 last= peak= cpu= oom= child=
sample() {
  local k v
  { read -r peak < "$cg/memory.peak"; } 2>/dev/null || peak=
  cpu=
  { while read -r k v; do [ "$k" = usage_usec ] && cpu=$v; done < "$cg/cpu.stat"; } 2>/dev/null
  oom=
  { while read -r k v; do [ "$k" = oom_kill ] && oom=$v; done < "$cg/memory.events"; } 2>/dev/null
  printf 'mdf-metrics/1 %s %s %s %s end\n' "${peak:--}" "${cpu:--}" "${oom:--}" "$n" >&7
}
finish() {
  trap '' TERM INT HUP USR1 USR2 PIPE ALRM
  [ -n "$child" ] && kill -KILL "$child" 2>/dev/null
  sample; printf '%s\n' "$1" >&8; exit "$1"
}
exec 9<> <(:)
if [ "$stat" = - ]; then
  exec 7> /dev/null 8> /dev/null
else
  set -C
  { exec 7> "$stat/metrics" 8> "$stat/exit"; } 2> /dev/null || exit 203
fi
trap 'finish 143' TERM INT HUP USR1 USR2 PIPE ALRM
if [ "$stat" = - ]; then
  exec 5>&1 6>&2
else
  { exec 5> "$stat/stdout" 6> "$stat/stderr"; } 2> /dev/null || exit 203
  set +C
fi
if [ -f "$ctl/stdin" ] && [ ! -L "$ctl/stdin" ]; then exec 4< "$ctl/stdin"; else exec 4< /dev/null; fi
export -n PWD OLDPWD 2>/dev/null
"$@" <&4 >&5 2>&6 4<&- 5>&- 6>&- 7>&- 8>&- 9>&- &
child=$!
exec 4<&- 5>&- 6>&-
while kill -0 "$child" 2>/dev/null; do
  read -r -t "$tick" -u 9 _
  n=$((n + 1))
  [ -e "$ctl/stop" ] && finish 202
  cur=
  { read -r cur < "$ctl/lease"; } 2>/dev/null
  if [ "$cur" != "$last" ]; then last=$cur; seen=$n; fi
  [ $((n - seen)) -gt "$lease" ] && finish 200
  [ "$n" -gt "$max" ] && finish 201
  [ $((n % 5)) -eq 0 ] && sample
done
wait "$child"; rc=$?
sample
printf '%s\n' "$rc" >&8
exit "$rc"
'''

_LINE = re.compile(rb"mdf-metrics/1 (\d{1,20}|-) (\d{1,20}|-) (\d{1,20}|-) (\d{1,12}) end")


def ticks(seconds, tick=TICK_SECONDS):
    """Whole ticks in `seconds`, at least one."""
    return max(1, int(round(seconds / tick)))


def entrypoint(max_seconds, lease_seconds, stat=True, ctl=CTL, stat_dir=STAT, cgroup=CGROUP, tick=TICK_SECONDS):
    """The entrypoint of a child: bash, the script and its six arguments; the command follows `--`."""
    return [BASH, "-c", SUPERVISOR, ARG0, str(ticks(max_seconds, tick)), str(ticks(lease_seconds, tick)), "%g" % tick,
            ctl, stat_dir if stat else "-", cgroup, "--"]


def last_metrics(data):
    """The last complete line of `stat/metrics`: {memory_peak_bytes, cpu_usec, oom_kill, tick}, a value the cgroup
    did not give is None; None without a complete line."""
    for line in reversed((data or b"").split(b"\n")):
        found = _LINE.fullmatch(line)
        if found:
            peak, cpu, oom, tick = (None if v == b"-" else int(v) for v in found.groups())
            return {"memory_peak_bytes": peak, "cpu_usec": cpu, "oom_kill": oom, "tick": tick}
    return None
