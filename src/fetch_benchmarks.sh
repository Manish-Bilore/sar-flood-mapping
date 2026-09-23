#!/usr/bin/env bash
# fetch_benchmarks.sh — start first; bandwidth (~1.6 MB/s) is the bottleneck. Resumable.
# 1) Sen1Floods11 hand-labelled (S1 + labels + splits)   ~ a few GB
# 2) UrbanSARFloods train/val tarball (HuggingFace)       size printed before download
set -eo pipefail
cd "$(dirname "$0")/.."; mkdir -p data/benchmarks && cd data/benchmarks
command -v gsutil >/dev/null || pip install -q gsutil
command -v aria2c >/dev/null || { echo "installing aria2 (parallel, resumable downloads)"; conda install -y -c conda-forge aria2 >/dev/null; }

echo ">> Sen1Floods11 v1.1 hand-labelled (cp -n skips existing files)"
S=gs://sen1floods11/v1.1
mkdir -p sen1floods11/splits
gsutil -m cp -n "$S/splits/flood_handlabeled/*" sen1floods11/splits/
for d in S1Hand LabelHand JRCWaterHand S1OtsuLabelHand; do
  mkdir -p sen1floods11/$d
  gsutil -m cp -n -r "$S/data/flood_events/HandLabeled/$d/*" sen1floods11/$d/
done
du -sh sen1floods11

echo ">> UrbanSARFloods (train/val)"
U=https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1/resolve/main/urban_sar_floods.tar.gz
HDR=$(curl -sIL --max-time 30 "$U" || true)
CODE=$(printf '%s\n' "$HDR" | awk '/^HTTP/{c=$2} END{print c}')
SZ=$(printf '%s\n' "$HDR" | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}')
echo "   HTTP ${CODE:-none}, size $((SZ/1024/1024)) MB"
if [[ "$CODE" == 401 || "$CODE" == 403 ]]; then
  echo "   gated dataset: accept terms at https://huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1"
  echo "   then: pip install -U huggingface_hub && huggingface-cli login   and re-run this script"
  H=$(python -c "from huggingface_hub import get_token;print(get_token() or '')" 2>/dev/null || true)
  [[ -n "$H" ]] && AUTH=(--header="Authorization: Bearer $H") || exit 0
fi
df -BG --output=avail . | tail -1
read -rp "   download now? [y/N] " a
[[ $a == [yY] ]] && aria2c -x8 -s8 -c --async-dns=false --max-tries=0 --retry-wait=10 "${AUTH[@]}" -d urbansarfloods -o urban_sar_floods.tar.gz "$U"
echo ">> done. Extract later: tar -xzf urbansarfloods/urban_sar_floods.tar.gz -C urbansarfloods"
