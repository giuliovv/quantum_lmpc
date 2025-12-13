#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"
S3_OUTPUT_URI="${S3_OUTPUT_URI:-}"

mkdir -p "$OUTPUT_DIR"

cmd=("$@")
if [ "${#cmd[@]}" -eq 0 ]; then
  cmd=(python3 -m duckrace.lmpc.duckietown_compare --iterations 15 --quantum --diagnostics --plot --no-augment)
fi

has_plot=0
has_plot_out=0
for token in "${cmd[@]}"; do
  case "$token" in
    --plot) has_plot=1 ;;
    --plot-out) has_plot_out=1 ;;
  esac
done

if [ "$has_plot" -eq 1 ] && [ "$has_plot_out" -eq 0 ]; then
  cmd+=(--plot-out "${OUTPUT_DIR%/}/lmpc_compare.png")
fi

echo "Running (workdir: $(pwd)):"
printf '  %q' "${cmd[@]}"
printf '\n'

set +e
xvfb-run -a -s "-screen 0 1280x720x24" "${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR%/}/run.log"
status="${PIPESTATUS[0]}"
set -e
if [ "$status" -ne 0 ]; then
  echo "Command failed with exit code $status" >&2
  exit "$status"
fi

if [ -n "$S3_OUTPUT_URI" ]; then
  echo "Syncing outputs to: $S3_OUTPUT_URI"
  aws s3 sync --only-show-errors "$OUTPUT_DIR" "$S3_OUTPUT_URI"
fi
