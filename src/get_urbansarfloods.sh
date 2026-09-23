#!/usr/bin/env bash
# get_urbansarfloods.sh — robust, resumable download of the 35 GiB UrbanSARFloods tarball.
# Why: last run went 9-10 MiB/s for 2 min, then all 8 connections stalled at once (local network drop),
#      and aria2 gave up after its timeout. This loop keeps restarting; aria2 -c resumes from the .aria2 control file.
#      The HF CDN URL is signed for 1 h; each restart re-resolves it, so expiry is handled too.
set -u
cd "$(dirname "$0")/../data/benchmarks" || exit 1
U=https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1/resolve/main/urban_sar_floods.tar.gz
O=urbansarfloods/urban_sar_floods.tar.gz
n=0
until [[ -f $O && ! -f $O.aria2 ]]; do
  n=$((n+1)); echo "[$(date +%T)] attempt $n"
  aria2c -x8 -s8 -k16M -c --async-dns=false --file-allocation=none \
         --timeout=60 --connect-timeout=30 --max-tries=0 --retry-wait=15 \
         --lowest-speed-limit=50K --summary-interval=300 --console-log-level=warn \
         -d urbansarfloods -o urban_sar_floods.tar.gz "$U" && break
  echo "[$(date +%T)] interrupted — waiting 60 s for the network, then resuming"; sleep 60
done
ls -l $O && echo "verifying gzip integrity (reads all 35 GiB, ~10 min)..." && gzip -t $O && echo "OK"
