#!/usr/bin/env bash
# setup_env.sh v2 — builds 'sarflood' env + a torch build that runs on Quadro P520 (sm_61)
# Changes vs v1:
#   - no mamba install into base (that silent solve of the anaconda base env was the hang);
#     conda >= 23.10 already uses the libmamba solver
#   - no `set -u` around conda (its activate scripts reference unset vars)
#   - every step prints; full log in setup_env.log
#   - resumable: re-running skips finished steps
set -eo pipefail
cd "$(dirname "$0")"
exec > >(tee -a setup_env.log) 2>&1
step(){ printf '\n[%(%H:%M:%S)T] >> %s\n' -1 "$*"; }

step "conda: $(conda --version), solver: $(conda config --show solver 2>/dev/null | awk '{print $2}')"
eval "$(conda shell.bash hook)"

ENV=sarflood
if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  step "env '$ENV' exists -> updating to match environment.yml (fixes a half-built env after Ctrl-C)"
  conda env update -n "$ENV" -f environment.yml --solver=libmamba
else
  step "creating env '$ENV' from environment.yml (5-15 min; progress below)"
  conda env create -f environment.yml --solver=libmamba
fi
conda activate "$ENV"
step "python: $(which python)  $(python --version)"

sm61_ok() {
python - <<'PY'
import sys, torch
arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
ok = "sm_61" in arch
if ok:
    x = torch.randn(1, 2, 64, 64, device="cuda"); w = torch.randn(4, 2, 3, 3, device="cuda")
    torch.nn.functional.conv2d(x, w).sum().item()      # forces a real kernel launch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| arch", arch, "|", "sm_61 OK" if ok else "sm_61 MISSING")
sys.exit(0 if ok else 1)
PY
}

if python -c "import torch" 2>/dev/null && sm61_ok; then
  step "torch already OK — skipping"
else
  step "installing torch (latest) + cu126  (~2.5 GB download)"
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
  if ! sm61_ok; then
    step "cu126 build lacks sm_61 -> installing torch 2.7.1 + cu118"
    pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
    sm61_ok || { step "GPU unusable — CPU training only"; }
  fi
fi

step "registering Jupyter kernel"
python -m ipykernel install --user --name "$ENV" --display-name "Python ($ENV)"

step "locating SNAP gpt"
if ! command -v gpt >/dev/null; then
  G=$(ls -d "$HOME"/esa-snap/bin/gpt "$HOME"/snap/bin/gpt /opt/esa-snap/bin/gpt /usr/local/esa-snap/bin/gpt 2>/dev/null | head -1 || true)
  [ -z "$G" ] && G=$(dirname "$(readlink -f "$(command -v snap)")")/gpt
  if [ -x "$G" ]; then
    grep -q "$(dirname "$G")" "$HOME/.bashrc" || echo "export PATH=\"$(dirname "$G"):\$PATH\"" >> "$HOME/.bashrc"
    step "gpt = $G (added to ~/.bashrc; open a new shell)"
  else
    step "gpt not found automatically — run: find / -name gpt -type f 2>/dev/null | head"
  fi
fi

step "import check"
python - <<'PY'
import importlib
for m in ["numpy","rasterio","xarray","rioxarray","geopandas","pystac_client","planetary_computer",
          "odc.stac","stackstac","asf_search","sarsen","segmentation_models_pytorch","timm","torch"]:
    try: print(f"OK   {m:28s} {getattr(importlib.import_module(m),'__version__','?')}")
    except Exception as e: print(f"FAIL {m:28s} {type(e).__name__}: {e}")
PY
step "done -> conda activate $ENV && python gpu_check.py && bash netcheck.sh"
