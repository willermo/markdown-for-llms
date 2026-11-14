#!/bin/bash
# Content manifest of the reference toolchain image of `mdf bench` (scripts/bench-toolchain/).
#
# Written once at build time to /etc/mdf-bench-toolchain.sha256, outside the enumerated trees, and
# printed again by `mdf bench prepare --toolchain`, which compares the two byte for byte. Its
# SHA-256 is `content_sha256` of markdown_for_llms/bench/data/toolchain.json: two builds from
# scratch give the same manifest, while the image ID changes with every build.
#
# Enumerated trees: /opt (the pixi environment, DjVuLibre, LibreOffice) and /usr (Debian
# libraries and fonts, this script), except every var/cache/fontconfig directory: fontconfig
# writes build-time timestamps in its caches (the one of the pixi environment is
# /opt/toolchain/.pixi/envs/default/var/cache/fontconfig), which are rebuilt from the font files,
# themselves enumerated. Everything else of Debian is identified by the package list. Not
# enumerated: /var, and /etc, which holds the manifest itself.
# Lines, each section in the byte order of the C locale:
#   f <sha256>  <path>        every regular file
#   l <path> -> <target>      every symbolic link
#   m <octal mode> <path>     every regular file and directory
#   p <package>=<version>     every installed Debian package
set -euo pipefail
export LC_ALL=C
roots=(/opt /usr)
walk() { find "${roots[@]}" -xdev -path '*/var/cache/fontconfig' -prune -o "$@"; }
if walk -name $'*[\n\t\\\\]*' -print -quit | grep -q .; then
  echo "content-manifest: a path with a newline, a tab or a backslash" >&2
  exit 1
fi
walk -type f -print0 | sort -z | xargs -0 -r sha256sum | sed 's/^/f /'
walk -type l -printf 'l %p -> %l\n' | sort
walk \( -type f -o -type d \) -printf 'm %m %p\n' | sort -k3
dpkg-query -W -f 'p ${Package}=${Version}\n' | sort
