# POSIX sh, sourced by the recipes of the Makefile and by bin/mdf (MFL051): what has to be known on the host
# before a container of the product is started. Nothing here reads a key: of the env file only the lines that
# set one of the names asked for are looked at, and the names asked for are settings of `make` and compose.
#
#   env_state FILE          the state of an env file on its ORIGINAL path, before any chmod or mount; one word:
#                           ok | missing | link | type | owner | mode
#   env_value FILE NAME     the value FILE gives NAME (the last line that sets it), without quotes; empty when unset.
#                           A comment is one only outside quotes: after a value between quotes, or after a space in
#                           a value without them. The value is text: nothing of it is ever run, and the function
#                           writes it only to its caller
#   docker_endpoint         sets mdf_socket and mdf_socket_gid from the endpoint of the Docker context in use;
#                           fails with the reason when it is not a local socket
#   docker_host_check       the requirements of the host: an ordinary user, Docker Engine 26 or newer (API 1.45),
#                           cgroup v2, not rootless; fails with the reason
#   compose_environment FILE   exports what compose.yaml interpolates: the name of the project (from the directory),
#                           the identity of the user, the Docker socket and its group, the folders of the
#                           archive and of the state, the port, and the settings of `make` and compose that
#                           FILE gives (MDF_COMPOSE_*)
#
#   gpu_host_check          the GPUs of the user (MFL051 B): an amd64 daemon with the runtime of the NVIDIA
#                           Container Toolkit; fails with the reason
#   server_environment FILE  the remote GPU server (compose.server.yaml): exports its address (never the loopback
#                           nor every address, in IPv4 or in IPv6: server_address_local),
#                           its port (23467) and the paths of its two private files, the TLS folder and the bearer
#                           key, each checked as a private regular file of the user; fails with the reason
#
#   install_cli CHECKOUT DIR   writes DIR/mdf: `#!/bin/sh`, a check that bin/mdf of CHECKOUT is still there (else
#                           "checkout not found: run make install-cli again", exit 127) and the `exec` of it
#   restart_report          one line for each service of the compose project that Docker started again by itself
#                           (`RestartCount` of `docker inspect`, or the state `restarting`): the service, the
#                           count and the last error line of its log. Returns 1 when there is one. The restart
#                           policy of the services stays as it is: a service that leaves with an error is started
#                           again, and this is how `make up` and `make status` say so.
#
# The link is checked here, on the host: inside a container the mounted file is already resolved, and the reader
# of the keys (`engine/envfile.py`, O_NOFOLLOW) could not see it. Limit: between this check and the mount a
# process of the same user can replace the file; it is the trust in the processes of the same user the product
# already has.

env_state() {
  if [ -L "$1" ]; then echo link; return; fi
  if [ ! -e "$1" ]; then echo missing; return; fi
  if [ ! -f "$1" ]; then echo type; return; fi
  # POSIX `ls -n`: mode, links, numeric owner. The name is not parsed.
  mdf_ls=$(LC_ALL=C ls -lnd -- "$1") || { echo type; return; }
  # shellcheck disable=SC2086
  set -- $mdf_ls
  if [ "$3" != "$(id -u)" ]; then echo owner; return; fi
  case $1 in -rw-------*) echo ok ;; *) echo mode ;; esac
}

env_value() {
  [ -f "$1" ] || return 0
  # In order (SOL-A2-N1): a value between single quotes, then one between double quotes, each up to its closing
  # quote and with nothing after it but a comment: what is between the quotes is the value, a ` #` in it included
  # (`t` ends there). Then a value without quotes: a comment starts at a `#` after a space. The last two lines
  # are for a value that opens a quote and holds that quote again: its outer quotes go, as before. A comment is
  # recognized only outside the quotes, and escape sequences are not handled: a backslash, and a quote after it,
  # stay as they are.
  sed -n "s/^[[:space:]]*\(export[[:space:]][[:space:]]*\)\{0,1\}$2[[:space:]]*=[[:space:]]*//p" "$1" | tail -n 1 \
    | sed -e "s/^'\([^']*\)'[[:space:]]*\(#.*\)\{0,1\}\$/\1/" -e t \
          -e 's/^"\([^"]*\)"[[:space:]]*\(#.*\)\{0,1\}$/\1/' -e t \
          -e 's/[[:space:]][[:space:]]*#.*$//' -e 's/[[:space:]]*$//' -e "s/^'\(.*\)'\$/\1/" -e 's/^"\(.*\)"$/\1/'
}

docker_endpoint() {
  mdf_endpoint=$(docker context inspect --format '{{.Endpoints.docker.Host}}') \
    || { echo "Docker does not answer: is it installed, and is the daemon running?" >&2; return 1; }
  case "$mdf_endpoint" in
    unix://*) mdf_socket=${mdf_endpoint#unix://} ;;
    *) echo "the Docker endpoint of the context in use is not a local socket: $mdf_endpoint" >&2; return 1 ;;
  esac
  [ -S "$mdf_socket" ] || { echo "the Docker socket of the context in use does not exist: $mdf_socket" >&2; return 1; }
  # the group of the socket itself (`-L`: Docker Desktop gives a link), numeric
  mdf_socket_gid=$(LC_ALL=C ls -lnL -- "$mdf_socket" | awk '{ print $4 }')
  case "$mdf_socket_gid" in ""|*[!0-9]*) echo "the group of the Docker socket cannot be read: $mdf_socket" >&2; return 1 ;; esac
}

docker_host_check() {
  [ "$(id -u)" != 0 ] || { echo "run as an ordinary user, not as root" >&2; return 1; }
  mdf_api=$(docker version --format '{{.Server.APIVersion}}') \
    || { echo "the Docker daemon does not answer to your user (docker version)" >&2; return 1; }
  mdf_major=${mdf_api%%.*}; mdf_minor=${mdf_api#*.}
  case "$mdf_major$mdf_minor" in ""|*[!0-9]*) echo "the version of the Docker API cannot be read: $mdf_api" >&2; return 1 ;; esac
  if [ "$mdf_major" -lt 1 ] || { [ "$mdf_major" -eq 1 ] && [ "$mdf_minor" -lt 45 ]; }; then
    echo "Docker Engine 26 or newer is needed (API 1.45): this daemon has API $mdf_api" >&2; return 1
  fi
  mdf_info=$(docker info --format '{{.CgroupVersion}} {{.SecurityOptions}}')
  case "$mdf_info" in "2 "*) ;; *) echo "cgroup v2 is needed: this daemon reports cgroup ${mdf_info%% *}" >&2; return 1 ;; esac
  case "$mdf_info" in *name=rootless*) echo "rootless Docker is not supported" >&2; return 1 ;; esac
}

compose_environment() {
  docker_endpoint || return 1
  mdf_top=$(pwd -P)
  # the name compose derives from the directory, said here so that `make` names the same project and volumes
  COMPOSE_PROJECT_NAME=$(basename "$mdf_top" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-' | sed 's/^[_-]*//')
  [ -n "$COMPOSE_PROJECT_NAME" ] || { echo "the name of this directory gives no name of a compose project" >&2; return 1; }
  MDF_COMPOSE_UID=$(id -u); MDF_COMPOSE_GID=$(id -g)
  MDF_COMPOSE_DOCKER_SOCKET=$mdf_socket; MDF_COMPOSE_DOCKER_GID=$mdf_socket_gid
  MDF_COMPOSE_ENV_FILE=$1
  case "$MDF_COMPOSE_ENV_FILE" in /*) ;; *) MDF_COMPOSE_ENV_FILE=$mdf_top/${MDF_COMPOSE_ENV_FILE#./} ;; esac
  for mdf_name in MDF_COMPOSE_DATA MDF_COMPOSE_STATE MDF_COMPOSE_MAX_CHILDREN MDF_COMPOSE_API_MEMORY \
      MDF_COMPOSE_WORKER_MEMORY MDF_COMPOSE_CLI_MEMORY MDF_COMPOSE_GPU_MEMORY MDF_PORT MDF_UPLOAD_MIB; do
    # the environment of `make` wins over the file, as for every variable of make
    eval "mdf_given=\${$mdf_name:-}"
    [ -n "$mdf_given" ] || mdf_given=$(env_value "$1" "$mdf_name")
    eval "$mdf_name=\$mdf_given"
  done
  : "${MDF_COMPOSE_DATA:=./data}" "${MDF_COMPOSE_STATE:=./state}" "${MDF_PORT:=23465}" "${MDF_UPLOAD_MIB:=50}"
  case "$MDF_COMPOSE_DATA" in /*) ;; *) MDF_COMPOSE_DATA=$mdf_top/${MDF_COMPOSE_DATA#./} ;; esac
  case "$MDF_COMPOSE_STATE" in /*) ;; *) MDF_COMPOSE_STATE=$mdf_top/${MDF_COMPOSE_STATE#./} ;; esac
  for mdf_name in MDF_PORT MDF_UPLOAD_MIB; do
    eval "mdf_given=\$$mdf_name"
    case "$mdf_given" in ""|*[!0-9]*) echo "$mdf_name in the env file is not a number" >&2; return 1 ;; esac
  done
  # a multipart upload is larger than its file: the API applies the limit of the file, nginx a little more
  MDF_COMPOSE_BODY_MIB=$((MDF_UPLOAD_MIB + 8))
  export MDF_COMPOSE_UID MDF_COMPOSE_GID MDF_COMPOSE_DOCKER_SOCKET MDF_COMPOSE_DOCKER_GID MDF_COMPOSE_ENV_FILE \
    MDF_COMPOSE_DATA MDF_COMPOSE_STATE MDF_COMPOSE_MAX_CHILDREN MDF_COMPOSE_API_MEMORY MDF_COMPOSE_WORKER_MEMORY \
    MDF_COMPOSE_CLI_MEMORY MDF_COMPOSE_GPU_MEMORY MDF_PORT MDF_COMPOSE_BODY_MIB COMPOSE_PROJECT_NAME
}

gpu_host_check() {
  # the GPUs of the user (MFL051 B): an amd64 daemon whose runtimes include the one of the NVIDIA Container Toolkit
  mdf_arch=$(docker version --format '{{.Server.Arch}}') || { echo "the Docker daemon does not answer" >&2; return 1; }
  [ "$mdf_arch" = amd64 ] || { echo "the local GPU and the remote server run on amd64 only: this daemon is $mdf_arch" >&2; return 1; }
  case "$(docker info --format '{{json .Runtimes}}' 2> /dev/null)" in
    *'"nvidia"'*) ;;
    *) echo "the NVIDIA Container Toolkit is not configured for this daemon (no runtime nvidia in docker info):" \
            "install it and run \`sudo nvidia-ctk runtime configure --runtime=docker\`, then restart Docker" >&2; return 1 ;;
  esac
}

# an IPv6 address written out as its eight groups (the two last ones from an IPv4 tail), then compared: "local"
# for the unspecified address, the loopback, and the IPv4 loopback or unspecified address mapped (five groups 0,
# a group ffff, then the IPv4 address, dotted or in two hexadecimal groups);
# nothing for any other address or a text that is not one. The text is read on the input, never run
mdf_ipv6_local='
function hex(s,   i, n) {
  if (s !~ /^[0-9a-f]+$/ || length(s) > 4) return -1
  n = 0
  for (i = 1; i <= length(s); i++) n = n * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1
  return n
}
NR == 1 {
  a = tolower($0)
  if (index(a, ".")) {
    t = a; sub(/.*:/, "", t)
    if (split(t, q, ".") != 4) exit
    for (i = 1; i <= 4; i++) if (q[i] !~ /^[0-9]+$/ || q[i] + 0 > 255) exit
    a = substr(a, 1, length(a) - length(t)) sprintf("%x:%x", q[1] * 256 + q[2], q[3] * 256 + q[4])
  }
  m = split(a, part, "::")
  if (m > 2) exit
  nl = (part[1] == "") ? 0 : split(part[1], l, ":")
  nr = 0
  if (m == 2) { if (part[2] != "") nr = split(part[2], r, ":"); if (nl + nr > 7) exit }
  else if (nl != 8) exit
  g = 0
  for (i = 1; i <= nl; i++) { v[++g] = hex(l[i]); if (v[g] < 0) exit }
  for (i = nl + nr; i < 8; i++) v[++g] = 0
  for (i = 1; i <= nr; i++) { v[++g] = hex(r[i]); if (v[g] < 0) exit }
  for (i = 1; i <= 5; i++) if (v[i] != 0) exit
  if (v[6] == 0 && v[7] == 0 && (v[8] == 0 || v[8] == 1)) print "local"
  if (v[6] == 65535 && ((v[7] >= 32512 && v[7] <= 32767) || (v[7] == 0 && v[8] == 0))) print "local"
}'

server_address_local() {   # ADDRESS: true when it is the loopback or every address, in IPv4 or in IPv6, in every
  # form compose takes: between brackets or not, groups with zeros or left out (::), an IPv4 tail, a zone (%...)
  mdf_address=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  mdf_address=${mdf_address#\[}; mdf_address=${mdf_address%\]}; mdf_address=${mdf_address%%\%*}
  case "$mdf_address" in
    ""|localhost|127.*|0.0.0.0) return 0 ;;
    *:*) [ "$(printf '%s\n' "$mdf_address" | LC_ALL=C awk "$mdf_ipv6_local")" = local ] ;;
    *) return 1 ;;
  esac
}

server_environment() {
  # the remote server (compose.server.yaml, MFL051 B): its address, its port and its two private files
  for mdf_name in MDF_COMPOSE_SERVER_ADDRESS MDF_COMPOSE_SERVER_PORT MDF_COMPOSE_SERVER_TLS MDF_COMPOSE_SERVER_KEY; do
    eval "mdf_given=\${$mdf_name:-}"
    [ -n "$mdf_given" ] || mdf_given=$(env_value "$1" "$mdf_name")
    eval "$mdf_name=\$mdf_given"
  done
  : "${MDF_COMPOSE_SERVER_PORT:=23467}" "${MDF_COMPOSE_SERVER_TLS:=./state/server/tls}" \
    "${MDF_COMPOSE_SERVER_KEY:=./state/server/key}"
  case "$MDF_COMPOSE_SERVER_PORT" in ""|*[!0-9]*) echo "MDF_COMPOSE_SERVER_PORT is not a number" >&2; return 1 ;; esac
  [ -n "$MDF_COMPOSE_SERVER_ADDRESS" ] || { echo "MDF_COMPOSE_SERVER_ADDRESS: the address of this machine on the LAN" \
    "or the VPS, where the product reaches it" >&2; return 1; }
  if server_address_local "$MDF_COMPOSE_SERVER_ADDRESS"; then
    echo "MDF_COMPOSE_SERVER_ADDRESS is not the loopback nor every address: the product reaches the server from" \
         "another machine" >&2; return 1
  fi
  for mdf_name in MDF_COMPOSE_SERVER_TLS MDF_COMPOSE_SERVER_KEY; do
    eval "mdf_given=\$$mdf_name"
    case "$mdf_given" in /*) ;; *) mdf_given=$mdf_top/${mdf_given#./} ;; esac
    eval "$mdf_name=\$mdf_given"
  done
  [ "$(env_state "$MDF_COMPOSE_SERVER_KEY")" = ok ] || { echo "MDF_COMPOSE_SERVER_KEY ($MDF_COMPOSE_SERVER_KEY):" \
    "a regular file of yours, mode 0600, with the key (\`umask 077; openssl rand -hex 32 > FILE\`)" >&2; return 1; }
  [ -f "$MDF_COMPOSE_SERVER_TLS/server.crt" ] && [ "$(env_state "$MDF_COMPOSE_SERVER_TLS/server.key")" = ok ] \
    || { echo "MDF_COMPOSE_SERVER_TLS ($MDF_COMPOSE_SERVER_TLS): server.crt and server.key, the key a regular file" \
              "of yours with mode 0600" >&2; return 1; }
  export MDF_COMPOSE_SERVER_ADDRESS MDF_COMPOSE_SERVER_PORT MDF_COMPOSE_SERVER_TLS MDF_COMPOSE_SERVER_KEY
}

install_cli() {    # CHECKOUT DIR: write DIR/mdf, a small file that runs bin/mdf of CHECKOUT (`make install-cli`)
  case "$1" in
    /*) ;;
    *) echo "install-cli: the checkout is given by its absolute path" >&2; return 1 ;;
  esac
  # the path is written between single quotes in a shell file: one that holds a quote or a line break is refused
  case "$1" in *"'"*) echo "install-cli: the path of this checkout holds a quote: move the checkout" >&2; return 1 ;; esac
  if [ "$(printf '%s' "$1" | wc -l)" -ne 0 ]; then
    echo "install-cli: the path of this checkout holds a line break: move the checkout" >&2; return 1
  fi
  [ -x "$1/bin/mdf" ] || { echo "install-cli: no bin/mdf in $1" >&2; return 1; }
  mkdir -p "$2" || return 1
  # the second line is how the installed file finds out that the checkout was moved or removed
  {
    printf '%s\n' '#!/bin/sh'
    printf '%s\n' "[ -x '$1/bin/mdf' ] || { echo \"checkout not found: run make install-cli again\" >&2; exit 127; }"
    printf '%s\n' "exec '$1/bin/mdf' \"\$@\""
  } > "$2/mdf.new" || return 1
  chmod 755 "$2/mdf.new" && mv -f "$2/mdf.new" "$2/mdf" || return 1
  echo "install-cli: $2/mdf runs $1/bin/mdf"
  case ":$PATH:" in *":$2:"*) ;; *) echo "install-cli: $2 is not on your PATH" ;; esac
}

restart_report() {
  mdf_restarted=0
  for mdf_id in $(docker ps -a -q --filter "label=com.docker.compose.project=$COMPOSE_PROJECT_NAME"); do
    mdf_line=$(docker inspect --format \
      '{{index .Config.Labels "com.docker.compose.service"}} {{.RestartCount}} {{.State.Status}}' "$mdf_id" 2> /dev/null) \
      || continue                               # removed between the list and this question
    # shellcheck disable=SC2086
    set -- $mdf_line
    case "${2:-}" in ""|*[!0-9]*) continue ;; esac
    if [ "$2" -eq 0 ] && [ "${3:-}" != restarting ]; then continue; fi
    mdf_restarted=1
    # the last line that names an error; without one, the last line that is not empty. One line, bounded.
    mdf_log=$(docker logs --tail 200 "$mdf_id" 2>&1 | tr -d '\r' | grep -v '^[[:space:]]*$' || true)
    mdf_last=$(printf '%s\n' "$mdf_log" | grep -i 'error' | tail -n 1)
    [ -n "$mdf_last" ] || mdf_last=$(printf '%s\n' "$mdf_log" | tail -n 1)
    mdf_last=$(printf '%s' "$mdf_last" | sed 's/^[[:space:]]*//' | cut -c1-300)
    echo "$1: started again by Docker $2 times (now ${3:-unknown}); the last error line of its log: ${mdf_last:-(no log)}"
  done
  [ "$mdf_restarted" -eq 0 ]
}
