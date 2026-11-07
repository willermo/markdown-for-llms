#!/usr/bin/env bash
# Full check of markdown-for-llms (AGENTS.md, "Verification"): installation preflight
# (T, O, P, S, B, reinstallation of the canonical wheel with a RECORD check, I, F), then the
# fast suite and tests/api. tests/packaging and tests/docker stay out: legacy contracts whose
# prerequisites were removed on 2026-10-17. Run it at the root of a clean checkout:
#   scripts/full-check.sh
# Paths come from the FULL_CHECK_* variables (docs/how-to/uv-environment.md, "Run the full
# check"): environment first, then the untracked .full-check.env of this checkout or of the
# main checkout. No download and no installation other than the wheel built here.
# Exit 0 only when every step and both suites pass.
set -euo pipefail
shopt -s inherit_errexit

stop() { printf 'STOP %s\n' "$*" >&2; exit 1; }
VARS=(FULL_CHECK_PYTHON_INSTALL_DIR FULL_CHECK_DEV_PYTHON FULL_CHECK_UV
      FULL_CHECK_UV_CACHE_DIR FULL_CHECK_BACKEND_WHEEL FULL_CHECK_TOKENIZER_CACHE)

# --- preconditions -------------------------------------------------------------------------
[ "$#" -eq 0 ] || stop "usage: scripts/full-check.sh (no arguments)"
top=$(git rev-parse --show-toplevel)
[ "$(pwd -P)" = "$top" ] || stop "not at the root of the checkout"
[ -z "$(git status --porcelain)" ] || stop "tree not clean"
head=$(git rev-parse HEAD)
for name in $(compgen -e); do
  if [[ $name == MDF_* ]]; then stop "MDF_ variables in the environment change the suites: $name"; fi
  # pytest reads PYTEST_ADDOPTS, PYTEST_PLUGINS, PYTEST_DISABLE_PLUGIN_AUTOLOAD and others that
  # change collection, selection or plugins; the full check needs none of them.
  if [[ $name == PYTEST_* ]]; then stop "PYTEST_ variables in the environment change the suites: $name"; fi
done

main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
env_file=
for f in "$top/.full-check.env" "$main/.full-check.env"; do
  if [ -f "$f" ]; then env_file=$f; break; fi
done
if [ -n "$env_file" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; *=*) ;; *) stop "$env_file: unknown line" ;; esac
    name=${line%%=*} value=${line#*=}
    case " ${VARS[*]} " in *" $name "*) ;; *) stop "$env_file: unknown line" ;; esac
    if [ -z "${!name+x}" ]; then printf -v "$name" '%s' "$value"; fi
  done < "$env_file"
fi
for name in "${VARS[@]}"; do
  value=${!name:-}
  value=${value%/}
  [ -n "$value" ] || stop "$name is not set (environment or .full-check.env)"
  case $value in /*) ;; *) stop "$name is not an absolute path" ;; esac
  printf -v "$name" '%s' "$value"
done
for name in FULL_CHECK_PYTHON_INSTALL_DIR FULL_CHECK_UV_CACHE_DIR FULL_CHECK_TOKENIZER_CACHE; do
  [ -d "${!name}" ] || stop "$name is not a directory"
done
[ -x "$FULL_CHECK_DEV_PYTHON" ] || stop "FULL_CHECK_DEV_PYTHON is not executable"
if [ ! -f "$FULL_CHECK_UV" ] || [ ! -x "$FULL_CHECK_UV" ] || [ -L "$FULL_CHECK_UV" ]; then
  stop "FULL_CHECK_UV is not an executable regular file"
fi
[ -f "$FULL_CHECK_BACKEND_WHEEL" ] || stop "FULL_CHECK_BACKEND_WHEEL is not a file"
MANAGED_DIR=$FULL_CHECK_PYTHON_INSTALL_DIR
MANAGED_PY="$MANAGED_DIR/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
[ -x "$MANAGED_PY" ] || stop "no managed CPython 3.12.13 in FULL_CHECK_PYTHON_INSTALL_DIR"
want=$(sed -n 's/^required-version = "==\(.*\)"$/\1/p' pyproject.toml)
have=$("$FULL_CHECK_UV" --version | cut -d' ' -f2)
if [ -z "$want" ] || [ "$have" != "$want" ]; then
  stop "FULL_CHECK_UV is uv $have, pyproject.toml requires $want"
fi

# The reinstallation replaces the package in the development venv: one full check at a time.
venv=$(cd "$(dirname "$FULL_CHECK_DEV_PYTHON")/.." && pwd -P)
exec 9<"$venv"
flock -n 9 || stop "another full check is using the development venv"

mkdir -p "$top/.cache/full-check"
PROOF="$top/.cache/full-check/$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$PROOF"
echo "$head" > "$PROOF/head.txt"
export TIKTOKEN_CACHE_DIR="$FULL_CHECK_TOKENIZER_CACHE" UV_CACHE_DIR="$FULL_CHECK_UV_CACHE_DIR"
export UV_PYTHON_INSTALL_DIR="$MANAGED_DIR" UV_PYTHON_DOWNLOADS=never UV_OFFLINE=1
D="$top/scripts/diagnostics/run-a001-fase0-uv"
echo "== full check at $head"
echo "evidence: $PROOF"
echo "paths: ${env_file:-environment only}"
echo "uv: $have"

# json_check <file> <python expression over d, R = managed directory>: prints ok or bad.
json_check() {
  "$MANAGED_PY" -I -B -c 'import json,sys
d=json.loads(open(sys.argv[1],encoding="utf-8").read())
print("ok" if eval(sys.argv[2],{"d":d,"R":sys.argv[3]}) else "bad")' "$1" "$2" "$MANAGED_DIR"
}

t=$SECONDS
lap() { echo "  ($((SECONDS - t))s)"; t=$SECONDS; }

# --- T: tokenizer cache ------------------------------------------------------------------
key=9b5ad71b2ce5302211f9c61530b329a4922fc6a4
sum=223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7
file="$TIKTOKEN_CACHE_DIR/$key"
[ "$(ls -A "$TIKTOKEN_CACHE_DIR")" = "$key" ] || stop "T: unexpected tokenizer cache content"
echo "$sum  $file" | sha256sum -c - > "$PROOF/T.out" || stop "T: sha256 mismatch"
[ "$(cat "$PROOF/T.out")" = "$file: OK" ] || stop "T: unexpected sha256sum output"
echo "T: ok"; lap

# --- O: origin of the managed interpreter ------------------------------------------------
"$MANAGED_PY" -I -B "$D/check_python_origin.py" --expected-managed "$MANAGED_DIR" > "$PROOF/O.out"
[ "$(json_check "$PROOF/O.out" 'd["isolated"] is True and d["expected_managed"] == R')" = ok ] || stop "O: unexpected output"
echo "O: ok"; lap

# --- P: input inventory ------------------------------------------------------------------
"$MANAGED_PY" -I -B "$D/make_input_inventory.py" --repo "$top" --managed "$MANAGED_DIR" \
  --backend-wheel "$FULL_CHECK_BACKEND_WHEEL" --uv "$FULL_CHECK_UV" --tokenizer-cache "$TIKTOKEN_CACHE_DIR" \
  --output "$PROOF/package-inputs.json" > "$PROOF/P.out"
[ "$(json_check "$PROOF/P.out" 'd["status"] == "PASS_INPUT_CAPTURE_ONLY"')" = ok ] || stop "P: unexpected output"
echo "P: $(cat "$PROOF/P.out")"; lap

# --- S: standalone source manifest -------------------------------------------------------
"$MANAGED_PY" -I -B "$D/make_source_manifest.py" --repo "$top" --standalone \
  --input-inventory "$PROOF/package-inputs.json" --uv "$FULL_CHECK_UV" --output "$PROOF/S.json" > "$PROOF/S.out"
[ "$(json_check "$PROOF/S.out" 'isinstance(d.get("id"), str) and d["id"] != ""')" = ok ] || stop "S: unexpected output"
echo "S: $(cat "$PROOF/S.out")"; lap

# --- B: offline build and audit of the archives ------------------------------------------
"$FULL_CHECK_UV" build --offline --python "$MANAGED_PY" --no-python-downloads \
  --build-constraints "$top/build-constraints.txt" --out-dir "$PROOF/dist" \
  > "$PROOF/B-build.out" 2> "$PROOF/B-build.err" || { tail -n 20 "$PROOF/B-build.err" >&2; stop "B: build failed"; }
# uv build also writes a one-byte `.gitignore` ("*") into the out dir.
listing=$(cd "$PROOF/dist" && LC_ALL=C ls -A | tr '\n' ' ')
[ "$listing" = ".gitignore markdown_for_llms-1.0.0-py3-none-any.whl markdown_for_llms-1.0.0.tar.gz " ] || stop "B: unexpected dist content: $listing"
[ "$(cat "$PROOF/dist/.gitignore")" = "*" ] || stop "B: unexpected dist .gitignore"
SDIST="$PROOF/dist/markdown_for_llms-1.0.0.tar.gz"
WHEEL="$PROOF/dist/markdown_for_llms-1.0.0-py3-none-any.whl"
(cd "$PROOF/dist" && sha256sum markdown_for_llms-1.0.0.tar.gz markdown_for_llms-1.0.0-py3-none-any.whl) > "$PROOF/dist.sha256"
"$FULL_CHECK_DEV_PYTHON" -I -B "$D/verify_distribution.py" --repo "$top" --source-manifest "$PROOF/S.json" \
  --sdist "$SDIST" --wheel "$WHEEL" --profile cpu --expected-managed "$MANAGED_DIR" --archives-only \
  --receipt "$PROOF/B.json" > "$PROOF/B.out"
[ "$(json_check "$PROOF/B.out" 'd["status"] == "PASS" and d["kind"] == "ARCHIVES"')" = ok ] || stop "B: unexpected output"
echo "B: $(cat "$PROOF/B.out")"; lap

# --- install: the wheel just built replaces the package in the development venv -----------
"$FULL_CHECK_UV" pip install --offline --python "$FULL_CHECK_DEV_PYTHON" --no-deps --no-build \
  --reinstall-package markdown-for-llms "$WHEEL" > "$PROOF/install.out" 2>&1 || { cat "$PROOF/install.out" >&2; stop "install failed"; }
echo "install: $(tr '\n' ' ' < "$PROOF/install.out")"
"$FULL_CHECK_DEV_PYTHON" -I -B -c 'import base64,csv,hashlib,importlib.metadata as m
dist=m.distribution("markdown-for-llms")
rows=list(csv.reader(dist.read_text("RECORD").splitlines()))
bad=[]
for path,digest,_ in rows:
    if not digest: continue
    algo,value=digest.split("=",1)
    data=dist.locate_file(path).read_bytes()
    if base64.urlsafe_b64encode(hashlib.new(algo,data).digest()).rstrip(b"=").decode()!=value: bad.append(path)
print("RECORD_OK" if not bad else "RECORD_BAD "+" ".join(bad), len(rows), dist.version, dist.locate_file(""))' > "$PROOF/install-record.out"
grep -q '^RECORD_OK ' "$PROOF/install-record.out" || stop "install: RECORD mismatch: $(cat "$PROOF/install-record.out")"
echo "install RECORD: $(cat "$PROOF/install-record.out")"; lap

# --- I: installation receipt -------------------------------------------------------------
"$FULL_CHECK_DEV_PYTHON" -I -B "$D/verify_distribution.py" --repo "$top" --source-manifest "$PROOF/S.json" \
  --sdist "$SDIST" --wheel "$WHEEL" --profile cpu --expected-managed "$MANAGED_DIR" --receipt "$PROOF/I.json" > "$PROOF/I.out"
[ "$(json_check "$PROOF/I.out" 'd["status"] == "PASS" and d["kind"] == "I"')" = ok ] || stop "I: unexpected output"
echo "I: $(cat "$PROOF/I.out")"; lap

# --- F: preflight ------------------------------------------------------------------------
"$FULL_CHECK_DEV_PYTHON" -I -B "$D/check_preflight.py" --repo "$top" --source "$PROOF/S.json" \
  --receipt "$PROOF/I.json" --standalone > "$PROOF/F.out"
[ "$(cat "$PROOF/F.out")" = PASS_CURRENT_I ] || stop "F: unexpected output"
echo "F: PASS_CURRENT_I"; lap

# --- suites ------------------------------------------------------------------------------
export RUN_SOURCE_MANIFEST="$PROOF/S.json" RUN_INSTALLATION_RECEIPT="$PROOF/I.json" RUN_TEST_MODE=standalone
# junit_counts <junit xml>: prints "tests failures errors skipped" of its single testsuite;
# exits nonzero, with the reason on stderr, when the file is missing, unreadable or unexpected.
junit_counts() {
  "$MANAGED_PY" -I -B -c 'import sys,xml.etree.ElementTree as E
try:
    r=E.parse(sys.argv[1]).getroot()
except (OSError,E.ParseError) as e:
    sys.exit(type(e).__name__+": "+str(e))
s=[r] if r.tag=="testsuite" else r.findall("testsuite") if r.tag=="testsuites" else []
if len(s)!=1: sys.exit("unexpected structure: root "+r.tag+", "+str(len(s))+" testsuite")
v=[s[0].get(k,"") for k in ("tests","failures","errors","skipped")]
if not all(x.isascii() and x.isdigit() for x in v): sys.exit("unexpected counts: "+repr(v))
print(*v)' "$1"
}
# A suite is green only when pytest exits 0 and its junit reads, counts at least one test and
# has no failures and no errors.
fail=()
run_suite() {                                         # run_suite <name> <pytest selection…>
  local name=$1 rc=0 why= counts tests failures errors skipped
  shift
  "$FULL_CHECK_DEV_PYTHON" -I -B -m pytest "$@" -q -rfE --junitxml "$PROOF/$name.xml" > "$PROOF/$name.out" 2>&1 || rc=$?
  echo "suite $name: pytest exit $rc; $(tail -n 1 "$PROOF/$name.out")"
  [ "$rc" -eq 0 ] || why="pytest exit $rc"
  if counts=$(junit_counts "$PROOF/$name.xml" 2> "$PROOF/$name.junit.err") \
     && [[ $counts =~ ^[0-9]+\ [0-9]+\ [0-9]+\ [0-9]+$ ]]; then
    read -r tests failures errors skipped <<< "$counts"
    echo "  junit: tests=$tests failures=$failures errors=$errors skipped=$skipped"
    if [ "$tests" -eq 0 ] || [ "$failures" -ne 0 ] || [ "$errors" -ne 0 ]; then
      why="${why:+$why, }junit without tests or with failures or errors"
    fi
  else
    echo "  junit: unreadable: $(tail -n 1 "$PROOF/$name.junit.err")"
    why="${why:+$why, }junit unreadable"
  fi
  if [ -n "$why" ]; then
    echo "  red: $why"
    tail -n 40 "$PROOF/$name.out" >&2
    fail+=("$name")
  fi
  lap
}
run_suite fast-suite tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker
run_suite api-suite tests/api

# --- end ---------------------------------------------------------------------------------
[ "$(git rev-parse HEAD)" = "$head" ] || fail+=("HEAD changed")
[ -z "$(git status --porcelain)" ] || fail+=("tree changed")
echo "total ${SECONDS}s"
if [ "${#fail[@]}" -eq 0 ]; then
  echo "FULL_CHECK_PASS $head"
else
  echo "FULL_CHECK_FAIL ${fail[*]}"
  exit 1
fi
