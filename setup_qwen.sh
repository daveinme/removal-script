#!/usr/bin/env bash
# Ambiente SEPARATO per il test esplorativo di Qwen Image Edit (GGUF).
#
# Va in un venv a parte (venv-qwen) perche' richiede diffusers/transformers
# molto piu' recenti di quelli usati da IOPaint nel venv principale (vedi
# nota in requirements-remote.txt): QwenImageTransformer2DModel e
# GGUFQuantizationConfig non esistono in diffusers==0.27.2.
#
# Uso sull'istanza, dopo il clone:
#   bash setup_qwen.sh [quant]        default quant: Q5_K_M
#
# Tempi/dimensioni attese (mostrati in chiaro durante l'esecuzione):
#   - venv + torch + diffusers/transformers: ~3-5 min, ~6-8 GB disco
#   - download transformer GGUF (Q5_K_M):    ~14 GB  (Q4_K_M ~11GB, Q8_0 ~19GB)
#   - download pipeline base (text_encoder+VAE): ~17-18 GB
#   In totale conta almeno 60-80 GB liberi e una connessione decente:
#   a 100 MB/s la sola parte modelli e' ~5-6 minuti, su reti piu' lente
#   anche 20-30 min — per questo ogni passo stampa un timestamp.
set -e

QUANT="${1:-Q5_K_M}"
T_START=$(date +%s)

fase() {
  local now=$(date +%s)
  echo
  echo "=== [$(date '+%H:%M:%S')] (+$((now - T_START))s totali) $1 ==="
}

fase "Setup ambiente Qwen Image Edit GGUF — quant $QUANT"

df -h . | awk 'NR==1 || NR==2 {print "  disco: "$0}'

# ------------------------------------------------------------- venv separato
if [ ! -d venv-qwen ]; then
  fase "Creo venv-qwen"
  python3 -m venv venv-qwen
fi
PY="./venv-qwen/bin/python"
PIP="./venv-qwen/bin/pip"
$PIP install -q --upgrade pip

if ! $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  fase "Installo torch (build cu121) — alcuni minuti"
  $PIP install torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121
fi

fase "Installo diffusers/transformers recenti + accelerate + gguf"
# diffusers >=0.31 ha QwenImageTransformer2DModel + GGUFQuantizationConfig;
# gguf serve al parser interno per leggere i file .gguf.
$PIP install -q "diffusers>=0.31.0" "transformers>=4.51.0" \
    accelerate huggingface-hub gguf pillow opencv-python-headless numpy

$PY - <<'PY'
import torch, diffusers
p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print(f"  torch {torch.__version__} | diffusers {diffusers.__version__}")
print(f"  GPU: {p.name} — {p.total_memory/1e9:.1f} GB" if p else "  ATTENZIONE: nessuna GPU rilevata")
PY

# --------------------------------------------------- prefetch pesi (con log)
fase "Scarico i pesi (transformer GGUF $QUANT + pipeline base) — questo e' il passo lungo"
echo "  Segui il progresso reale (velocita', ETA) qui sotto:"
HF_HUB_ENABLE_HF_TRANSFER=0 $PY - <<PY
import time
from huggingface_hub import hf_hub_download, snapshot_download

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] --> transformer GGUF $QUANT (~14 GB per Q5_K_M)")
hf_hub_download(
    "unsloth/Qwen-Image-Edit-2511-GGUF",
    "qwen-image-edit-2511-$QUANT.gguf",
)
print(f"[{time.strftime('%H:%M:%S')}] transformer scaricato in {time.time()-t0:.0f}s")

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] --> pipeline base Qwen-Image-Edit-2509 (text_encoder+VAE, ~17-18 GB)")
snapshot_download(
    "Qwen/Qwen-Image-Edit-2509",
    allow_patterns=["text_encoder/*", "vae/*", "scheduler/*", "tokenizer/*",
                    "processor/*", "*.json", "*.txt"],
)
print(f"[{time.strftime('%H:%M:%S')}] pipeline base scaricata in {time.time()-t0:.0f}s")
PY

fase "Spazio occupato dalla cache HuggingFace"
du -sh ~/.cache/huggingface/hub 2>/dev/null | sed 's/^/  /'
df -h . | awk 'NR==1 || NR==2 {print "  disco: "$0}'

T_END=$(date +%s)
fase "Fatto in $((T_END - T_START))s totali"
echo
echo "  Test:  ./venv-qwen/bin/python src/qwen_test.py <nome_capo> --quant $QUANT"
echo
