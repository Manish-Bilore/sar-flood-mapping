#!/usr/bin/env bash
# run_events.sh — CPU/network part for all remaining events (safe to run while the GPU trains).
# nohup bash src/run_events.sh > events.log 2>&1 &
cd "$(dirname "$0")/.."
EV="assam_2020 assam_2022 bihar_2020 bengaluru_2022 delhi_2023 hyderabad_2020 mumbai_2024"
for e in $EV; do
  echo "######## $e  $(date +%T)"
  python -u src/fetch_event_rtc.py $e   || { echo "!! rtc failed: $e"; continue; }
  python -u src/fetch_ancillary.py $e   || { echo "!! ancillary failed: $e"; continue; }
  for co in data/events/$e/rtc/co_*.tif; do
    tag=$(basename $co .tif | cut -d_ -f2)
    python -u src/baseline_classical.py $e --co $tag || echo "!! baseline failed: $e $tag"
  done
done
python -u src/qa_rtc.py
bash src/disk_report.sh | head -8
echo "######## done $(date +%T)"
