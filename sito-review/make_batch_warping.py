#!/usr/bin/env python3
"""
Batch di warping automatico + confronto a TRE colonne:

    PRIMA (scatto)  |  NOSTRO warping  |  DOPO (Photoshop, a mano)

Il senso: finora sul sito c'erano solo scatto e ritocco umano, cioe' il
riferimento. Questo script aggiunge in mezzo cio' che produce il nostro
codice, che e' l'unico modo per dire se siamo vicini o lontani.

Regola che questo script rispetta: si applicano parametri ragionevoli e si
pubblica QUELLO CHE ESCE, senza ritoccare i valori capo per capo finche' non
somigliano al riferimento. Altrimenti il confronto non dimostra nulla.

Uso: python3 make_batch_warping.py && python3 sync_r2.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scout_warp import applica_linea                    # noqa: E402
from scout_warp.rileva import linea_vita                # noqa: E402

SRC = ROOT.parent / "postproduzione" / "originale"
OUT = ROOT / "output" / "iterazioni" / "v17_warping_automatico"

H = 1700
GAP = 30
LABEL_H = 104
_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(size, bold=False):
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(_FONT_DIR / n), size)
    except OSError:
        return ImageFont.load_default()


def carica(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    alpha = im[:, :, 3].copy() if im.ndim == 3 and im.shape[2] == 4 else None
    rgb = im[:, :, :3].copy()
    if alpha is not None:
        rgb[alpha < 128] = 255
    return rgb, alpha


def bgr2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def scala(img: Image.Image, h: int):
    return img.resize((max(1, round(img.width * h / img.height)), h), Image.LANCZOS)


def elabora(stem: str):
    """Rileva la linea, applica il warp, restituisce (originale, nostro, diag)."""
    rgb, alpha = carica(SRC / "PRIMA" / f"{stem}.png")
    p0, p1, diag = linea_vita(rgb, alpha)

    if not diag["ok"]:
        return rgb, None, diag

    h, w = rgb.shape[:2]
    # l'orlo resta fermo: si corregge la vita, non si trascina tutto il capo
    congelate = [{"x0": 0, "y0": h * 0.60, "x1": w, "y1": h}]
    out, _, info = applica_linea(rgb, p0, p1, alpha, congelate=congelate)
    diag = {**diag, **info}
    return rgb, out, diag


def confronto(stem: str, rgb, nostro, diag):
    dopo_path = SRC / "DOPO" / f"{stem}.jpg"
    dopo = Image.open(dopo_path).convert("RGB") if dopo_path.exists() else None

    col = [scala(bgr2pil(rgb), H)]
    tit = ["PRIMA (scatto)"]
    if nostro is not None:
        col.append(scala(bgr2pil(nostro), H))
        tit.append("NOSTRO warping (automatico)")
    else:
        vuoto = Image.new("RGB", (int(H * 0.6), H), "#f2f2f4")
        d = ImageDraw.Draw(vuoto)
        d.text((20, H // 2 - 20), "nessuna correzione\n(vedi motivo sopra)",
               fill="#888", font=_font(28))
        col.append(vuoto)
        tit.append("NOSTRO warping — non applicato")
    if dopo is not None:
        col.append(scala(dopo, H))
        tit.append("DOPO (Photoshop, a mano)")

    w_tot = sum(c.width for c in col) + GAP * (len(col) - 1)
    canvas = Image.new("RGB", (w_tot, H + LABEL_H), "white")
    x = 0
    d = ImageDraw.Draw(canvas)
    for c, t in zip(col, tit):
        canvas.paste(c, (x, LABEL_H))
        d.text((x + 10, 8), t, fill="#111", font=_font(30, bold=True))
        x += c.width + GAP

    stato = diag.get("motivo", "")
    colore = "#0a7d2e" if diag.get("ok") else "#a33"
    d.text((10, 48), f"{stem}", fill="#111", font=_font(26, bold=True))
    d.text((10, 76), stato[:150], fill=colore, font=_font(24))
    # Avvertenza onesta: qui il warp gira sullo scatto GREZZO, quindi la
    # gruccia si deforma insieme al capo (si vede piegarsi). Nella pipeline
    # vera il warp viene dopo la rimozione e il problema non si pone.
    d.text((canvas.width - 10, 8),
           "warp applicato allo scatto grezzo: la gruccia si deforma col capo",
           fill="#a33", font=_font(22), anchor="ra")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{stem}_tre.jpg"
    canvas.save(p, quality=90)
    return p


def main():
    stems = sorted({p.stem for p in (SRC / "PRIMA").glob("*.png")}
                   & {p.stem for p in (SRC / "DOPO").glob("*.jpg")})
    stems = [s for s in stems if s.lower() != "thumbs"]
    print(f"{len(stems)} capi con PRIMA e DOPO\n")

    for stem in stems:
        try:
            rgb, nostro, diag = elabora(stem)
            p = confronto(stem, rgb, nostro, diag)
            stato = "applicato" if nostro is not None else "SALTATO"
            print(f"{stem:<18} {stato:<10} {diag.get('motivo','')[:70]}")
            print(f"{'':<18} -> {p.name} ({p.stat().st_size//1024} KB)")
        except Exception as e:
            print(f"{stem:<18} ERRORE {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
