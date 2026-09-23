#!/usr/bin/env bash
# disk_report.sh — free space + what the project and the usual caches occupy
cd "$(dirname "$0")/.."
echo "== filesystem";            df -h . | tail -1
echo "== project data";          du -sh data/benchmarks/* data/events/* 2>/dev/null | sort -h
echo "== partial downloads";     ls -l data/benchmarks/urbansarfloods/ 2>/dev/null
echo "== reclaimable caches (safe to clear)"
du -sh ~/anaconda3/pkgs ~/.cache/pip ~/.cache/huggingface ~/.cache/torch 2>/dev/null
echo "   clear with: conda clean -a -y ; pip cache purge"
echo "== largest dirs in \$HOME (top 12)"; du -xh --max-depth=2 ~ 2>/dev/null | sort -h | tail -12
