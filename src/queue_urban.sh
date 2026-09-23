#!/usr/bin/env bash
# queue_urban.sh — waits for the Sen1Floods11 benchmark to release the GPU, then runs the urban experiment.
# nohup bash src/queue_urban.sh > train_usf.log 2>&1 &
cd "$(dirname "$0")/.."
while pgrep -f train_s1f11.py >/dev/null; do sleep 300; done
echo "[$(date +%T)] GPU free — starting UrbanSARFloods runs"
for b in co_int int all; do
  python -u src/train_usf.py --model unet_r34 --bands $b --epochs 25 --samples 3000
done
echo "[$(date +%T)] urban queue done"
