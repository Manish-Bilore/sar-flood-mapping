#!/usr/bin/env bash
# sysprobe.sh — capability check for the SAR flood-mapping project
# Usage: bash sysprobe.sh | tee sysprobe_$(hostname).txt
set -u
hr(){ printf '\n==== %s ====\n' "$1"; }

hr "OS / kernel";      lsb_release -ds 2>/dev/null || cat /etc/os-release | head -2; uname -r
hr "CPU";              lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|MHz' 
hr "RAM / swap";       free -h
hr "Disk (home, /tmp)"; df -h "$HOME" /tmp | awk 'NR==1||!/tmpfs/'
hr "GPU (PCI)";        lspci | grep -Ei 'vga|3d|display'
hr "NVIDIA driver/VRAM"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
  nvidia-smi | grep -i 'CUDA Version'
else echo "nvidia-smi not found (no driver or no NVIDIA GPU)"; fi

hr "Python / conda"
command -v python3 && python3 --version
command -v conda && conda --version
command -v mamba && mamba --version

hr "Python libs"
python3 - <<'EOF'
import importlib
mods = ["torch","torchvision","segmentation_models_pytorch","timm","rasterio","xarray",
        "rioxarray","stackstac","odc.stac","pystac_client","planetary_computer",
        "asf_search","sarsen","geopandas","numpy","dask"]
for m in mods:
    try:
        mod = importlib.import_module(m); print(f"OK   {m:28s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"MISS {m:28s} ({type(e).__name__})")
try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU:", p.name, f"{p.total_memory/1e9:.1f} GB", "cc", f"{p.major}.{p.minor}")
except Exception: pass
EOF

hr "GDAL / SAR tools"
command -v gdalinfo && gdalinfo --version
for t in gpt snap isce2 topsStack.py gmtsar.csh R Rscript quarto docker; do
  printf '%-14s ' "$t"; command -v "$t" >/dev/null && echo found || echo missing
done

hr "Network reachability (HTTP status)"
for u in https://planetarycomputer.microsoft.com/api/stac/v1 \
         https://stac.dataspace.copernicus.eu/v1 \
         https://api.daac.asf.alaska.edu/services/search/param?maxResults=1 \
         https://bhoonidhi.nrsc.gov.in ; do
  printf '%-70s ' "$u"; curl -s -o /dev/null -w '%{http_code}\n' --max-time 10 "$u"
done

hr "Download throughput (~100 MB test)"
curl -s -o /dev/null -w 'avg %{speed_download} B/s\n' --max-time 30 \
  https://speed.cloudflare.com/__down?bytes=100000000 || echo "speed test failed"
