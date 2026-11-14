#!/usr/bin/env bash
# Sequential stages that depend on each other. Each stage is spawned detached; this script
# only polls the volume, so a dropped connection costs a retry, not the run.
#   bash orchestrate.sh qwen     # chain generations 2-4 (after es,chaing1)
#   bash orchestrate.sh llama    # second family: teachers, then students
cd "$(dirname "$0")"
wait_done() {  # stage fam max_minutes
  local t=0
  while [ $t -lt "$3" ]; do
    out=$(modal run modal_app.py --stage "$1" --fam "$2" --stage-status 2>/dev/null | grep "cells done" | cut -c1-40)
    echo "$(date +%H:%M) $1 [$2] $out"
    done_n=$(echo "$out" | sed -E 's#^([0-9]+)/([0-9]+).*#\1#'); tot=$(echo "$out" | sed -E 's#^([0-9]+)/([0-9]+).*#\2#')
    [ -n "$done_n" ] && [ "$done_n" = "$tot" ] && return 0
    sleep 180; t=$((t+3))
  done
  echo "$(date +%H:%M) $1 [$2] not complete after $3 min; continuing"; return 1
}
launch() {  # stage fam workers
  for try in 1 2 3; do
    out=$(modal run --detach modal_app.py --stage "$1" --fam "$2" --workers "$3" --spawn 2>&1 | grep -c spawned)
    echo "$(date +%H:%M) launched $1 [$2]: $out chunks"
    [ "$out" -gt 0 ] && return 0
    sleep 60
  done
  return 1
}
if [ "$1" = "qwen" ]; then
  wait_done es,chaing1 qwen25 90
  for g in 2 3 4; do launch chaing$g qwen25 6; wait_done chaing$g qwen25 60; done
elif [ "$1" = "llama" ]; then
  t=0; until modal run modal_app.py --stage addon --fam llama31 --dry-run 2>/dev/null | grep -q "teacher records missing\|cells build" && \
        modal run modal_app.py --stage teachers --fam llama31 --dry-run 2>/dev/null | grep -q "7/7 cells build"; do
    echo "$(date +%H:%M) waiting for llama base pass"; sleep 180; t=$((t+3)); [ $t -gt 90 ] && exit 1; done
  launch teachers llama31 7; wait_done teachers llama31 60
  launch addon llama31 8; wait_done addon llama31 90
fi
echo "$(date +%H:%M) orchestrate $1 finished"
