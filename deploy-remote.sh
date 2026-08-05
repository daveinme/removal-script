#!/usr/bin/env bash
# Prepara e sincronizza il prototipo su un'istanza GPU remota (Vast, VPS...).
#
# Serve a testare i modelli che sulla 3060 vanno in out-of-memory (LDM, MAT,
# ZITS via IOPaint) e FLUX.2 Klein, mai provato.
#
# NON servono webhook ne' API pubbliche: la GUI e' gia' una web app locale,
# la si raggiunge dal proprio browser tramite tunnel SSH.
#
# Uso:
#   ./deploy-remote.sh sync    utente@host -p PORTA   copia codice + capi di test
#   ./deploy-remote.sh setup   utente@host -p PORTA   crea il venv e installa
#   ./deploy-remote.sh tunnel  utente@host -p PORTA   apre il tunnel sulla GUI
#   ./deploy-remote.sh pull    utente@host -p PORTA   riporta indietro i risultati
set -euo pipefail

CMD="${1:-}"; shift || true
HOST="${1:-}"; shift || true
SSH_OPTS=("$@")                       # es. -p 2703  oppure  -i chiave.pem

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCOUT_ROOT="$(dirname "$LOCAL_ROOT")"
REMOTE_DIR="~/scout-gruccia"

if [[ -z "$CMD" || -z "$HOST" ]]; then
  sed -n '2,20p' "$0"; exit 1
fi

case "$CMD" in

sync)
  echo "==> Copio codice e immagini di test su $HOST:$REMOTE_DIR"
  # esclude venv, pesi dei modelli (si riscaricano) e output: sarebbero GB
  rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
    --exclude 'venv/' --exclude 'output/' --exclude 'modelli/' \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '.env' \
    "$LOCAL_ROOT/" "$HOST:$REMOTE_DIR/"

  echo "==> Copio i capi di test (STYLESHOOT)"
  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p $REMOTE_DIR/capi_test"
  rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
    "$SCOUT_ROOT/postproduzione/IMMAGINI STYLESHOOT/" \
    "$HOST:$REMOTE_DIR/capi_test/"
  echo "Fatto. Ora:  ./deploy-remote.sh setup $HOST ${SSH_OPTS[*]}"
  ;;

setup)
  echo "==> Creo l'ambiente su $HOST (qualche minuto: scarica torch e i modelli)"
  ssh "${SSH_OPTS[@]}" "$HOST" bash -s <<'REMOTE'
set -e
cd ~/scout-gruccia
python3 -m venv venv 2>/dev/null || true
./venv/bin/pip install -q --upgrade pip

if ! ./venv/bin/python -c "import torch" 2>/dev/null; then
  echo "torch assente: installo build cu121"
  ./venv/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
fi
./venv/bin/pip install -q -r requirements-remote.txt

echo "--- GPU disponibile ---"
./venv/bin/python - <<'PY'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} — {p.total_memory/1e9:.1f} GB")
else:
    print("NESSUNA GPU RILEVATA")
PY

mkdir -p modelli/sam3
if [ ! -s modelli/sam3/sam3.pt ]; then
  echo "--- scarico SAM 3 (3.3 GB) ---"
  ./venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('1038lab/sam3','sam3.pt',local_dir='modelli/sam3')"
fi
echo "Ambiente pronto."
REMOTE
  echo "Ora:  ./deploy-remote.sh tunnel $HOST ${SSH_OPTS[*]}"
  ;;

tunnel)
  echo "==> Avvio la GUI su $HOST e apro il tunnel"
  echo "    Apri nel browser:  http://localhost:8095"
  echo "    (Ctrl+C per chiudere)"
  ssh "${SSH_OPTS[@]}" -L 8095:localhost:8095 "$HOST" \
    "cd ~/scout-gruccia && ./venv/bin/python gui/app.py"
  ;;

pull)
  echo "==> Riporto i risultati da $HOST"
  mkdir -p "$LOCAL_ROOT/output/remoto"
  rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" \
    "$HOST:$REMOTE_DIR/output/" "$LOCAL_ROOT/output/remoto/"
  echo "Scaricati in output/remoto/"
  ;;

*)
  echo "Comando sconosciuto: $CMD"; sed -n '2,20p' "$0"; exit 1
  ;;
esac
