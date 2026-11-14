#!/usr/bin/env bash
# Development tool (Linux with a systemd user manager): run a command under a memory cap.
#   scripts/run-capped.sh NAME MAX_MEMORY -- COMMAND [ARGS...]
# MAX_MEMORY as systemd accepts it, for example 8G. The command runs in a transient systemd user
# scope with MemoryMax and no swap, so a runaway process is killed inside the group instead of the
# machine running out of memory (the full check, long bench runs, diagnoses). systemd does not
# expand variables in the command (--expand-environment=no). Appends the start and end time, the
# exit status, the peak memory of the whole group (cgroup memory.peak, oom kills) and the largest
# single process (GNU time) to .cache/run-capped/NAME.mem at the root of the checkout (ignored by
# git); the command's own output is untouched. Exits with the command's status.
# Containers started through the Docker daemon (the bench toolchain) are not in this scope: the
# bench caps each of them with its own --memory.
set -u
if [ "$#" -lt 4 ] || [ "$3" != "--" ]; then
  echo "usage: scripts/run-capped.sh NAME MAX_MEMORY -- COMMAND [ARGS...]" >&2
  exit 2
fi
name=$1 max=$2
shift 3
case $name in ''|*/*|.*) echo "run-capped: NAME must be a plain file name" >&2; exit 2 ;; esac
top=$(git rev-parse --show-toplevel) || exit 2
log="$top/.cache/run-capped/$name.mem"
mkdir -p "$(dirname "$log")"
start=$(date +%FT%T%z)
systemd-run --user --scope --quiet --expand-environment=no -p MemoryMax="$max" -p MemorySwapMax=0 \
  -p OOMPolicy=continue -- bash -c '
  log=$1; shift
  /usr/bin/time -f "time: maxrss_kb %M elapsed %e s exit %x" -o "$log.time" "$@"; rc=$?
  cg=$(cut -d: -f3 /proc/self/cgroup)
  echo "group_peak_bytes $(cat "/sys/fs/cgroup$cg/memory.peak") group_max $(cat "/sys/fs/cgroup$cg/memory.max")" \
    "oom_kill $(grep oom_kill "/sys/fs/cgroup$cg/memory.events" | cut -d" " -f2)" > "$log.group"
  exit $rc' _ "$log" "$@"
rc=$?
{ echo "start $start end $(date +%FT%T%z) exit $rc"; cat "$log.group" "$log.time" 2>/dev/null; } >> "$log"
rm -f "$log.group" "$log.time"
tail -n 3 "$log" >&2
exit $rc
