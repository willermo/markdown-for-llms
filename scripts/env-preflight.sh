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
  # are for a value that opens a quote and holds that quote again: its outer quotes go, as before. It is the
  # reading of the parser of the product (python-dotenv) for the values a path or a number can have.
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
      MDF_COMPOSE_WORKER_MEMORY MDF_COMPOSE_CLI_MEMORY MDF_PORT MDF_UPLOAD_MIB; do
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
    MDF_COMPOSE_CLI_MEMORY MDF_PORT MDF_COMPOSE_BODY_MIB COMPOSE_PROJECT_NAME
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
