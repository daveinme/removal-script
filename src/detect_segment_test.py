#!/usr/bin/env python3
"""
Test di detection (GroundingDINO) + segmentazione (SAM2) della gruccia
su un'immagine reale, per validare la pipeline prima di costruire la UI.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from groundingdino.util.inference import load_model, load_image, predict, annotate

DINO_CONFIG = str(
    ROOT / "venv/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DINO_WEIGHTS = str(ROOT / "modelli/groundingdino/groundingdino_swint_ogc.pth")

PROMPT = (
    "clothes hanger . hook . metal hook . wire hook . hanger bar . swivel hook "
    ". plastic clip . hanger clip . black clip . garment clip"
)
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def detect_hanger(image_path: Path):
    model = load_model(DINO_CONFIG, DINO_WEIGHTS)
    model = model.to(DEVICE)
    image_source, image = load_image(str(image_path))

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )
    return image_source, boxes, logits, phrases


def segment_with_sam2(image_source: np.ndarray, boxes_xyxy: np.ndarray):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_ckpt = str(ROOT / "modelli/sam2/sam2.1_hiera_large.pt")
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_source)

    masks, scores, _ = predictor.predict(
        box=boxes_xyxy,
        multimask_output=False,
    )
    return masks, scores


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 detect_segment_test.py <path immagine> [tag_versione]")
        print("  tag_versione: es. 'v3_dilate10' -> salva in output/iterazioni/v3_dilate10/")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    out_dir = (ROOT / "output" / "iterazioni" / tag) if tag else (ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Detection gruccia su {image_path.name} (device={DEVICE})...")
    image_source, boxes, logits, phrases = detect_hanger(image_path)
    print(f"  trovate {len(boxes)} box (grezze): {phrases} (score={[round(float(l), 2) for l in logits]})")

    # Filtro di sanità: scarta box con area eccessiva (falsi positivi su
    # sfondo/intera immagine) — una gruccia reale non supera mai ~25% del
    # frame in un prodotto shot verticale.
    MAX_BOX_AREA_FRACTION = 0.25
    keep = [i for i in range(len(boxes)) if float(boxes[i, 2] * boxes[i, 3]) <= MAX_BOX_AREA_FRACTION]
    dropped = len(boxes) - len(keep)
    if dropped:
        print(f"  scartate {dropped} box troppo grandi (area > {MAX_BOX_AREA_FRACTION*100:.0f}% immagine)")
    boxes = boxes[keep]
    logits = logits[keep]
    phrases = [phrases[i] for i in keep]
    print(f"  rimaste {len(boxes)} box valide: {phrases} (score={[round(float(l), 2) for l in logits]})")

    if len(boxes) == 0:
        print("Nessuna gruccia rilevata dopo il filtro, controlla prompt/soglie.")
        return

    annotated = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(str(out_dir / f"{image_path.stem}_boxes.png"), annotated)
    print(f"  salvato: {out_dir / (image_path.stem + '_boxes.png')}")

    h, w, _ = image_source.shape
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    boxes_xyxy = boxes_xyxy.numpy()

    print(f"[2/3] Segmentazione SAM2 su {len(boxes_xyxy)} box...")
    masks, scores = segment_with_sam2(image_source, boxes_xyxy)
    print(f"  maschere ottenute: {masks.shape}, scores={scores}")

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        m2d = m.squeeze() if m.ndim > 2 else m
        combined_mask |= (m2d > 0.5).astype(np.uint8) * 255

    cv2.imwrite(str(out_dir / f"{image_path.stem}_mask.png"), combined_mask)
    print(f"  salvato: {out_dir / (image_path.stem + '_mask.png')}")

    overlay = image_source.copy()
    overlay[combined_mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(image_source, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(out_dir / f"{image_path.stem}_overlay.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"[3/3] salvato overlay: {out_dir / (image_path.stem + '_overlay.png')}")


if __name__ == "__main__":
    main()
