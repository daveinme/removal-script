#!/bin/bash
set -e
source venv/bin/activate
cd "$(dirname "$0")"
TAG="v10_batch_completo"
FILES=(BRM10465-M-AA CMC10778-M-AA FLP10848-W-86 GNN10258-W-OF JNS10416-M-UC TSH11514-W-AD TSH11617-M-AN VST10509-W-VE)
for stem in "${FILES[@]}"; do
  echo "=== $stem ==="
  img=$(ls "../postproduzione/originale/PRIMA/${stem}."* 2>/dev/null | head -1)
  if [ -z "$img" ]; then
    echo "MANCA: $stem"
    continue
  fi
  python3 src/detect_segment_test.py "$img" "$TAG" 2>&1 | tail -20
  python3 src/inpaint_test.py "$stem" "$TAG" 2>&1 | tail -20
  echo ""
done
echo "BATCH COMPLETATO"
