#!/bin/sh
# The `init` service of compose (MFL051): init-volumes.sh UID GID DIR...
#
# A named volume is born of root. For the root of each volume, never below it: when it is already of the user
# with mode 0700, nothing is done (every `make up` after the first, every `make restart`); otherwise the root
# alone is taken (`chown 0:0`, which CAP_CHOWN allows), given its mode by its owner (so no CAP_FOWNER is needed)
# and handed to the user. Two runs in a row leave the same state. A path that is not a real directory is refused.
#
# It runs as UID 0 with the capability CHOWN alone, without network and with a read-only root: the only container
# of the product with a capability. CHOWN, CHMOD and STAT name the commands (the tests replace them).
set -eu
[ "$#" -ge 2 ] || { echo "usage: init-volumes.sh UID GID DIR..." >&2; exit 2; }
uid=$1 gid=$2
shift 2
for number in "$uid" "$gid"; do
  case "$number" in ""|*[!0-9]*) echo "init: UID and GID are numbers" >&2; exit 2 ;; esac
done
[ "$uid" != 0 ] || { echo "init: the volumes are never given to UID 0" >&2; exit 2; }
: "${CHOWN:=chown}" "${CHMOD:=chmod}" "${STAT:=stat}"
for dir in "$@"; do
  if [ -L "$dir" ] || [ ! -d "$dir" ]; then echo "init: not a directory: $dir" >&2; exit 1; fi
  state=$($STAT -c '%u %g %a' "$dir")
  if [ "$state" = "$uid $gid 700" ]; then echo "init: $dir ok"; continue; fi
  $CHOWN 0:0 "$dir"
  $CHMOD 0700 "$dir"
  $CHOWN "$uid:$gid" "$dir"
  echo "init: $dir prepared (was $state)"
done
