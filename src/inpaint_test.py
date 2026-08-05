#!/usr/bin/env python3
"""
Test di inpainting mascherato (LaMa) sulla maschera gruccia già calcolata da
detect_segment_test.py. Compositing: solo i pixel dentro la maschera (dilatata
di pochi px) vengono sostituiti, il resto resta l'originale 1:1.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


WHITE_BG_THRESHOLD = 245  # pixel con tutti i canali >= questo valore = sfondo piatto


BG_EXTRA_DILATE_PX = 45  # dilatazione extra SOLO sulle componenti già classificate BG


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 inpaint_test.py <nome_base> [tag_versione] [dilate_px]")
        print("  tag_versione: es. 'v7_split' -> legge/scrive in output/iterazioni/v7_split/")
        print("  dilate_px: dilatazione della maschera, per antialiasing bordo (default 6)")
        sys.exit(1)

    stem = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    dilate_px = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out_dir = (ROOT / "output" / "iterazioni" / tag) if tag else (ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{stem}_mask.png"

    orig_candidates = list((ROOT.parent / "postproduzione/originale/PRIMA").glob(f"{stem}.*"))
    if not orig_candidates:
        print(f"Originale non trovato per {stem}")
        sys.exit(1)
    orig_path = orig_candidates[0]

    print(f"Originale: {orig_path}")
    print(f"Maschera: {mask_path}")

    original = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if original.shape[:2] != mask.shape[:2]:
        print(f"ATTENZIONE: shape diverse originale={original.shape[:2]} mask={mask.shape[:2]}, resize mask")
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    print(f"Dilatazione maschera (antialiasing bordo): {dilate_px}px")
    dilate_kernel = np.ones((dilate_px, dilate_px), np.uint8)
    mask_dilated = cv2.dilate(mask, dilate_kernel, iterations=1)
    cv2.imwrite(str(out_dir / f"{stem}_mask_dilated.png"), mask_dilated)

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Split per COMPONENTE CONNESSA (non per singolo pixel): un oggetto
    # scuro (gancio metallico) sospeso su sfondo bianco ha pixel interni
    # non-bianchi, ma va comunque riempito di bianco — LaMa fatica su
    # oggetti isolati ad alto contrasto perché ne "preserva" il bordo netto
    # invece di cancellarlo (verificato: lascia il gancio quasi intatto).
    # Criterio: guarda l'ANELLO attorno a ciascuna componente. Se è
    # prevalentemente sfondo bianco, l'intera componente -> fill diretto.
    # Altrimenti (componente adiacente/dentro il capo) -> LaMa.
    is_bg = np.all(original_rgb >= WHITE_BG_THRESHOLD, axis=2)
    n_labels, labels = cv2.connectedComponents((mask_dilated > 0).astype(np.uint8))
    ring_kernel = np.ones((31, 31), np.uint8)

    mask_bg = np.zeros(mask_dilated.shape, dtype=bool)
    mask_fg = np.zeros(mask_dilated.shape, dtype=bool)
    for label in range(1, n_labels):
        comp = (labels == label)
        comp_u8 = comp.astype(np.uint8) * 255
        ring = (cv2.dilate(comp_u8, ring_kernel) > 0) & ~comp
        if ring.sum() == 0:
            mask_fg |= comp
            continue
        bg_fraction = is_bg[ring].mean()
        decision = "BG" if bg_fraction >= 0.9 else "FG"
        ys_c, xs_c = np.where(comp)
        print(
            f"  componente {label}: area={comp.sum()} bbox=x[{xs_c.min()}:{xs_c.max()}] "
            f"y[{ys_c.min()}:{ys_c.max()}] bg_fraction={bg_fraction:.3f} -> {decision}"
        )
        if bg_fraction >= 0.9:
            mask_bg |= comp
        else:
            mask_fg |= comp
    print(f"Split maschera: {mask_bg.sum()} px su sfondo (fill diretto), {mask_fg.sum()} px su capo (LaMa)")

    # Dilatazione extra SOLO sulle componenti BG (isolate su sfondo puro per
    # definizione): copre bordi sottili non catturati da SAM2 (es. il
    # profilo di un gancio dove si piega/intreccia) senza il rischio delle
    # v3-v6, che dilatavano l'intero frame e fondevano componenti FG vicine
    # a elementi reali (etichette, cuciture) — qui quelle restano intatte
    # perché già segregate in mask_fg prima di questo passo.
    if mask_bg.sum() > 0:
        bg_extra_kernel = np.ones((BG_EXTRA_DILATE_PX, BG_EXTRA_DILATE_PX), np.uint8)
        mask_bg = cv2.dilate(mask_bg.astype(np.uint8) * 255, bg_extra_kernel) > 0
        mask_bg &= ~mask_fg  # non invadere mai l'area già assegnata a LaMa
        print(f"Dopo dilatazione extra BG ({BG_EXTRA_DILATE_PX}px): {mask_bg.sum()} px")

    final_rgb = original_rgb.copy()
    final_rgb[mask_bg] = [255, 255, 255]

    if mask_fg.sum() > 0:
        mask_fg_u8 = (mask_fg.astype(np.uint8)) * 255
        # LaMa elabora l'intero frame con una rete conv full-image: su foto
        # da 20+ Mpx satura la VRAM. Si ritaglia un crop attorno alla sola
        # componente "capo" della maschera (con margine di contesto) e si
        # reinserisce solo quella patch nell'originale a piena risoluzione.
        ys, xs = np.where(mask_fg_u8 > 0)
        margin = 250
        y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin)
        x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin)
        print(f"Crop regione LaMa: x[{x0}:{x1}] y[{y0}:{y1}] ({x1-x0}x{y1-y0}px)")

        crop_img = final_rgb[y0:y1, x0:x1]
        crop_mask = mask_fg_u8[y0:y1, x0:x1]

        # Pre-fill con cv2.inpaint (Telea, classico non-AI) PRIMA di LaMa:
        # elimina il bordo ad alto contrasto dell'oggetto (es. gancio scuro
        # su tessuto chiaro) sostituendolo con una stima di colore/texture
        # locale grezza. LaMa riceve così un'area già "vuota" da rifinire,
        # invece di un bordo netto che tende a preservare invece di
        # cancellare (verificato: causa gli oggetti rimasti visibili).
        print("Pre-fill cv2.inpaint (Telea) prima di LaMa...")
        crop_img_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
        prefilled_bgr = cv2.inpaint(crop_img_bgr, crop_mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
        crop_img = cv2.cvtColor(prefilled_bgr, cv2.COLOR_BGR2RGB)

        print("Eseguo inpainting LaMa sul crop (può richiedere qualche secondo)...")
        from simple_lama_inpainting import SimpleLama

        lama = SimpleLama()
        result_crop = np.array(lama(Image.fromarray(crop_img), Image.fromarray(crop_mask)))

        # LaMa arrotonda internamente le dimensioni (multipli di 8): riallinea
        # l'output esattamente alla shape del crop prima del compositing.
        ch, cw = crop_img.shape[:2]
        if result_crop.shape[:2] != (ch, cw):
            result_crop = cv2.resize(result_crop, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        crop_mask_bool = (crop_mask > 0)[:, :, None]
        final_rgb[y0:y1, x0:x1] = np.where(crop_mask_bool, result_crop, crop_img)
    else:
        print("Nessun pixel su capo da inpaintare, solo fill diretto su sfondo.")

    out_path = out_dir / f"{stem}_inpainted.png"
    Image.fromarray(final_rgb).save(out_path)
    print(f"Salvato: {out_path} (risoluzione invariata: {final_rgb.shape[1]}x{final_rgb.shape[0]})")


if __name__ == "__main__":
    main()
