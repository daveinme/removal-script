#!/usr/bin/env python3
"""
Motore Qwen Image Edit (GGUF): logica condivisa tra qwen_test.py (test
esplorativo da riga di comando) e la GUI (gui/app.py, engine="qwen").

Vive in un venv SEPARATO (venv-qwen) perche' richiede diffusers/transformers
piu' recenti di quelli usati da IOPaint nel venv principale della GUI — le
due dipendenze non possono coesistere nello stesso processo Python. Per
questo la GUI non importa questo modulo direttamente: lo invoca come
subprocess con l'interprete di venv-qwen (vedi run_qwen_subprocess in
gui/app.py), passando immagine e maschera come file temporanei e leggendo
il risultato allo stesso modo — stesso principio di isolamento gia'
previsto nel progetto per FLUX.2 Klein (vedi requirements-remote.txt).

Uso diretto da riga di comando (dentro venv-qwen):
  python3 qwen_engine.py <originale.png> <mask.png> <output.png> \
      [--quant Q5_K_M] [--steps 30] [--garment felpa] [--margin 220]
"""
import argparse
import sys
import time

import cv2
import numpy as np
from PIL import Image

REPO_GGUF = "unsloth/Qwen-Image-Edit-2511-GGUF"
BASE_PIPELINE = "Qwen/Qwen-Image-Edit-2509"  # text_encoder/VAE/scheduler: presi da qui

# Contesto extra intorno alla mask: il modello vede il tessuto circostante
# (pieghe, cuciture, illuminazione) per dedurre come continuarlo sotto la
# gruccia, ma il compositing finale tocca SOLO i pixel della edit mask.
CONTEXT_MARGIN_PX = 220
EDIT_MASK_DILATE_PX = 8   # poco: la mask deve restare stretta sulla gruccia
FEATHER_PX = 12           # sfuma il bordo della sostituzione, come fill_v3()

PROMPT_TEMPLATE = (
    "Remove the clothes hanger completely. Reconstruct the fabric of the "
    "{garment} exactly as it would look without any hanger, preserving the "
    "same color, texture, weave, folds, seams and lighting of the "
    "surrounding fabric. Do not change anything else in the image."
)
NEGATIVE_PROMPT = (
    "hanger, hook, different color, different texture, new garment, "
    "cartoon, illustration, blurry, distorted seams"
)

_PIPE_CACHE = {}  # {quant: pipeline} — evita ricaricare se si chiama piu' volte nello stesso processo


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def carica_pipeline(quant: str):
    if quant in _PIPE_CACHE:
        return _PIPE_CACHE[quant]

    import torch
    from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel, QwenImageEditPipeline

    gguf_file = f"qwen-image-edit-2511-{quant}.gguf"
    gguf_url = f"https://huggingface.co/{REPO_GGUF}/resolve/main/{gguf_file}"

    t0 = time.time()
    log(f"Carico transformer GGUF ({quant}) da {REPO_GGUF} — download se non in cache locale HF...")
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_url,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
    )
    log(f"Transformer pronto in {time.time()-t0:.1f}s")

    t0 = time.time()
    log(f"Carico pipeline base ({BASE_PIPELINE}) con transformer sostituito...")
    pipe = QwenImageEditPipeline.from_pretrained(
        BASE_PIPELINE,
        transformer=transformer,
        dtype=torch.bfloat16,
    )
    # Il text encoder (Qwen2.5-VL, ~16GB bf16) serve solo per codificare il
    # prompt: lo si tiene su GPU solo per quel passo, poi si scarica, cosi'
    # transformer GGUF (~11-14GB) + VAE restano soli in VRAM durante i passi
    # di diffusione. Necessario per stare sotto 24GB con Q5_K_M o piu' pesanti.
    pipe.enable_model_cpu_offload()
    log(f"Pipeline pronta in {time.time()-t0:.1f}s (cpu offload attivo)")
    _PIPE_CACHE[quant] = pipe
    return pipe


def prepara_crop(original_bgr, mask_gray, margin_px):
    """Ritaglia un'area di contesto intorno alla mask (bbox + margine),
    arrotondata a multipli di 16 (requisito tipico dei modelli diffusion)."""
    ys, xs = np.where(mask_gray > 0)
    if len(xs) == 0:
        raise ValueError("Maschera vuota")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    h, w = mask_gray.shape[:2]
    x0 = max(0, x0 - margin_px)
    y0 = max(0, y0 - margin_px)
    x1 = min(w, x1 + margin_px)
    y1 = min(h, y1 + margin_px)

    def espandi_16(lo, hi, lim):
        cur = hi - lo
        resto = (-cur) % 16
        hi = min(lim, hi + resto)
        if hi - lo < cur + resto:  # non c'era spazio a destra/sotto, sposta lo
            lo = max(0, lo - (cur + resto - (hi - lo)))
        return lo, hi

    x0, x1 = espandi_16(x0, x1, w)
    y0, y1 = espandi_16(y0, y1, h)

    return original_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def componi(original_bgr, generated_rgb_crop, edit_mask_crop, box):
    """result = original ovunque, generated SOLO dentro edit_mask (con
    feathering sul bordo) — mai un semplice replace dell'intera immagine."""
    x0, y0, x1, y1 = box
    generated_bgr = cv2.cvtColor(np.array(generated_rgb_crop), cv2.COLOR_RGB2BGR)
    if generated_bgr.shape[:2] != (y1 - y0, x1 - x0):
        generated_bgr = cv2.resize(generated_bgr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)

    mask_dilated = cv2.dilate(edit_mask_crop, np.ones((EDIT_MASK_DILATE_PX, EDIT_MASK_DILATE_PX), np.uint8))
    mask_soft = cv2.GaussianBlur(mask_dilated, (0, 0), FEATHER_PX).astype(np.float32) / 255.0
    mask_soft = mask_soft[..., None]

    result = original_bgr.copy()
    crop_orig = result[y0:y1, x0:x1].astype(np.float32)
    crop_new = generated_bgr.astype(np.float32)
    blended = crop_orig * (1 - mask_soft) + crop_new * mask_soft
    result[y0:y1, x0:x1] = blended.astype(np.uint8)
    return result


def rimuovi_gruccia(original_bgr, mask_gray, quant="Q5_K_M", steps=30,
                     garment="felpa", margin_px=CONTEXT_MARGIN_PX):
    """Entry point richiamabile: originale BGR (np.ndarray) + maschera gray
    -> immagine BGR risultato, compositata solo dentro la maschera."""
    if mask_gray.shape[:2] != original_bgr.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (original_bgr.shape[1], original_bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    crop_bgr, box = prepara_crop(original_bgr, mask_gray, margin_px)
    x0, y0, x1, y1 = box
    edit_mask_crop = mask_gray[y0:y1, x0:x1]
    log(f"Crop di contesto: {crop_bgr.shape[1]}x{crop_bgr.shape[0]}px (box={box})")

    pipe = carica_pipeline(quant)
    prompt = PROMPT_TEMPLATE.format(garment=garment)
    log(f"Prompt: {prompt}")

    crop_rgb_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    t0 = time.time()
    log(f"Genero — {steps} step, quant {quant}...")
    out = pipe(
        image=crop_rgb_pil,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=steps,
        true_cfg_scale=4.0,
    ).images[0]
    log(f"Generazione completata in {time.time()-t0:.1f}s")

    return componi(original_bgr, out, edit_mask_crop, box)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("originale", help="path immagine originale (PNG/JPG)")
    ap.add_argument("mask", help="path maschera grayscale (PNG, 255=gruccia)")
    ap.add_argument("output", help="path PNG di output")
    ap.add_argument("--quant", default="Q5_K_M")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--garment", default="felpa")
    ap.add_argument("--margin", type=int, default=CONTEXT_MARGIN_PX)
    args = ap.parse_args()

    original_bgr = cv2.imread(args.originale, cv2.IMREAD_COLOR)
    if original_bgr is None:
        log(f"ERRORE: impossibile leggere {args.originale}")
        sys.exit(1)
    mask_gray = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        log(f"ERRORE: impossibile leggere {args.mask}")
        sys.exit(1)

    result = rimuovi_gruccia(original_bgr, mask_gray, quant=args.quant,
                              steps=args.steps, garment=args.garment,
                              margin_px=args.margin)
    cv2.imwrite(args.output, result)
    log(f"Salvato: {args.output}")


if __name__ == "__main__":
    main()
