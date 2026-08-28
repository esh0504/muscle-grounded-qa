#!/bin/bash
# Release entrypoint: verify ArtiSynth is usable, ensure /data layout, then exec.
set -e

# Everything this container writes is world-writable (dirs 777 / files 666),
# so host accounts never fight over ownership of generated data.
umask 000

AH="${ARTISYNTH_HOME:-/opt/artisynth/artisynth_core}"
NATIVE="$AH/lib/Linux64"

if [ ! -d "$AH/classes" ]; then
  echo "[setup] ERROR: ArtiSynth classes missing at $AH"
  exit 1
fi

# Native solver libs are fetched at build time; refetch once if the layer was
# built without network access to the ArtiSynth file server.
if [ ! -d "$NATIVE" ] || [ -z "$(ls -A "$NATIVE" 2>/dev/null)" ]; then
  echo "[setup] Linux native libs missing -> fetching (one-time fallback)..."
  if [ ! -w "$AH/lib" ]; then
    echo "[setup] ERROR: $AH/lib not writable (running as $(id -u)) — rebuild the image with network access"
    exit 1
  fi
  if ( cd "$AH" && java -cp "lib/vfs2.jar:bin/libraryInstaller.jar" \
         artisynth.core.driver.LibraryInstaller -updateLibs \
         -remoteSource https://www.artisynth.org/files/lib/ ); then
    echo "[setup] native libs ready in $NATIVE"
  else
    echo "[setup] ERROR: native lib fetch failed (need network)"
    exit 1
  fi
fi

if ! mkdir -p /data/mesh /data/index /data/qa 2>/dev/null; then
  echo "[setup] ERROR: /data is not writable by uid $(id -u) (DATA_DIR on the host)."
  echo "[setup] Create it as your own account first:  mkdir -p <DATA_DIR>"
  echo "[setup] If it already exists root-owned:      sudo chown -R \$(id -u):\$(id -g) <DATA_DIR>"
  echo "[setup] Quick alternative (anyone can write): sudo chmod -R 777 <DATA_DIR>"
  exit 1
fi
if [ -f /opt/artisynth/REFS ]; then
  echo "[setup] ArtiSynth refs: $(tr '\n' ' ' < /opt/artisynth/REFS)"
fi
echo "[setup] pipeline=/workspace  data=/data  config=$PIPELINE_CONFIG"

exec "$@"
