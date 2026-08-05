#!/usr/bin/env bash
# Ricostruisce l'ambiente completo su un'istanza GPU (Vast, RunPod, VPS).
#
# Uso sull'istanza, dopo il clone:
#   git clone https://github.com/daveinme/removal-script.git
#   cd removal-script && bash setup_vast.sh
#
# Poi si avvia la GUI e la si raggiunge dal proprio browser via tunnel SSH
# (vedi in fondo). Non servono API pubbliche ne' webhook.
set -e

echo "==================================================================="
echo " Setup pipeline rimozione gruccia — istanza GPU"
echo "==================================================================="

# ---------------------------------------------------------------- GPU
python3 - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} — {p.total_memory/1e9:.1f} GB")
    else:
        print("ATTENZIONE: nessuna GPU rilevata")
except ImportError:
    print("torch non ancora installato (verra' installato ora)")
PY

# ------------------------------------------------------------- ambiente
if [ ! -d venv ]; then
  echo "==> Creo il virtualenv"
  python3 -m venv venv
fi
PY_BIN="./venv/bin/python"
PIP="./venv/bin/pip"
$PIP install -q --upgrade pip

if ! $PY_BIN -c "import torch" 2>/dev/null; then
  echo "==> Installo torch (build cu121)"
  $PIP install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121
fi

echo "==> Installo le dipendenze"
$PIP install -q -r requirements-remote.txt

# --------------------------------------------------------------- SAM 3
mkdir -p modelli/sam3
if [ ! -s modelli/sam3/sam3.pt ]; then
  echo "==> Scarico SAM 3 (3.3 GB)"
  # mirror non-gated: facebook/sam3 richiede l'approvazione manuale
  $PY_BIN -c "
from huggingface_hub import hf_hub_download
hf_hub_download('1038lab/sam3','sam3.pt',local_dir='modelli/sam3')"
else
  echo "==> SAM 3 gia' presente"
fi

# ---------------------------------------------------------------- ZITS
if [ ! -d modelli/zits_repo ]; then
  echo "==> Clono ZITS"
  git clone --depth 1 https://github.com/DQiaole/ZITS_inpainting.git modelli/zits_repo
  # il codice e' del 2022: due patch per numpy/scikit-image moderni
  sed -i 's|from skimage.measure import compare_ssim|from skimage.metrics import structural_similarity as compare_ssim|' \
      modelli/zits_repo/src/inpainting_metrics.py
  grep -rl 'np\.float\b\|np\.int\b\|np\.bool\b' --include=*.py modelli/zits_repo \
      | xargs -r sed -i -e 's/np\.float\b/float/g' -e 's/np\.int\b/int/g' -e 's/np\.bool\b/bool/g'
  echo "    NOTA: i pesi ZITS (4 file) vanno scaricati a mano da Google Drive"
  echo "          in modelli/zits_repo/ckpt/ — vedi README."
fi

# ---------------------------------------------- modelli IOPaint (erase)
echo "==> Precarico i modelli IOPaint (MAT, MIGAN, FcF)"
echo "    Su 24 GB dovrebbero girare tutti, anche quelli che sulla 3060"
echo "    andavano in out-of-memory."
for M in mat migan fcf; do
  $PY_BIN - <<PY 2>/dev/null | tail -1
import torch, numpy as np
from iopaint.model import models
from iopaint.schema import InpaintRequest, HDStrategy
try:
    m = models['$M'](device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    img = np.full((512,512,3),200,np.uint8)
    msk = np.zeros((512,512),np.uint8); msk[200:300,200:300]=255
    m(img, msk, InpaintRequest(hd_strategy=HDStrategy.CROP,
      hd_strategy_crop_trigger_size=1280, hd_strategy_crop_margin=196,
      hd_strategy_resize_limit=2048))
    print('    $M OK')
    del m; torch.cuda.empty_cache()
except Exception as e:
    print('    $M FALLITO:', str(e)[:90])
PY
done

echo
echo "==================================================================="
echo " Pronto."
echo
echo " Avvia la GUI:      ./venv/bin/python gui/app.py"
echo
echo " Dal TUO pc, apri il tunnel e poi http://localhost:8095 :"
echo "   ssh -p <PORTA> -L 8095:localhost:8095 root@<HOST>"
echo
echo " Le immagini si caricano con drag&drop dal browser."
echo "==================================================================="
