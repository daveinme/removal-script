#!/usr/bin/env python3
"""
Test esplorativo: Qwen Image Edit (GGUF quantizzato) come motore di fill per
il caso che LaMa/patch_fill non reggono — occlusioni grandi su zone semantiche
(colletto, spalle) dove serve "capire" la struttura del capo, non solo
propagare texture locale.

Wrapper a riga di comando su qwen_engine.py (logica condivisa con la GUI,
dove Qwen e' selezionabile come motore "qwen" — vedi inpaint_qwen() in
gui/app.py). Usa questo script quando vuoi testare rapidamente su un capo
gia' processato con detect_segment_test.py/inpaint_test.py, senza passare
dalla GUI.

Gira in un venv SEPARATO (venv-qwen): diffusers 0.27.2 del progetto
principale e' troppo vecchio per QwenImageTransformer2DModel/GGUF.
Vedi setup_qwen.sh.

Uso:
  ./venv-qwen/bin/python src/qwen_test.py <nome_base> [--quant Q5_K_M] [--steps 30]

Richiede che sia gia' stata generata la maschera con detect_segment_test.py
(stessa convenzione di inpaint_test.py: output/<stem>_mask.png).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qwen_engine import rimuovi_gruccia, log, CONTEXT_MARGIN_PX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="nome base, es. MGL10708-M-BE")
    ap.add_argument("--quant", default="Q5_K_M",
                     help="quant GGUF (Q4_K_M piu' leggero ~11GB, Q5_K_M ~14GB, Q8_0 ~19GB)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--garment", default="felpa", help="descrizione del capo per il prompt")
    ap.add_argument("--margin", type=int, default=CONTEXT_MARGIN_PX)
    ap.add_argument("--tag", default=None, help="sottocartella output/iterazioni/<tag>/")
    args = ap.parse_args()

    out_dir = (ROOT / "output" / "iterazioni" / args.tag) if args.tag else (ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{args.stem}_mask.png"

    orig_candidates = list((ROOT.parent / "postproduzione/originale/PRIMA").glob(f"{args.stem}.*"))
    if not orig_candidates:
        log(f"ERRORE: originale non trovato per {args.stem}")
        sys.exit(1)
    orig_path = orig_candidates[0]

    log(f"Originale: {orig_path}")
    log(f"Maschera:  {mask_path}")
    if not mask_path.exists():
        log("ERRORE: maschera assente. Genera prima con detect_segment_test.py")
        sys.exit(1)

    original_bgr = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    result = rimuovi_gruccia(original_bgr, mask_gray, quant=args.quant,
                              steps=args.steps, garment=args.garment,
                              margin_px=args.margin)

    stamp = time.strftime("%H%M%S")
    out_path = out_dir / f"{args.stem}_qwen_{args.quant}_{stamp}.png"
    cv2.imwrite(str(out_path), result)
    log(f"Salvato: {out_path}")


if __name__ == "__main__":
    main()
