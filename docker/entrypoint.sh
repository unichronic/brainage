#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    exec python -m fastbrainage --help
fi

case "$1" in
    preprocess)
        shift
        exec /opt/fastbrainage/docker/preprocess_fastspm.sh "$@"
        ;;
    extract-features|train|predict|model-info)
        exec python -m fastbrainage "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
