#!/usr/bin/env python3
"""
Genera i confronti PRIMA/DOPO (ritocco manuale in Photoshop) per il task
warping, dentro output/iterazioni/ cosi' il sito di review li pubblica come
una sezione a se'.

Servono a decidere QUALI capi hanno davvero bisogno del warp: il "dopo" e'
lavoro umano, non output nostro, quindi e' il riferimento da imitare.

Le due versioni hanno proporzioni diverse (il PRIMA e' lo scatto grezzo, il
DOPO e' gia' su canvas 2000x3000), quindi si normalizza l'altezza: cosi' le
inclinazioni sono confrontabili a occhio.

Uso: python3 make_confronti_warping.py && python3 sync_r2.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
SRC = ROOT.parent / "postproduzione" / "originale"
OUT = ROOT / "output" / "iterazioni" / "v16_warping_riferimento"

H = 2000          # altezza comune: sul sito si apre a piena risoluzione
GAP = 40
LABEL_H = 96      # spazio per due righe di testo leggibili

# Il font bitmap di default di PIL e' latin-1 (niente trattini lunghi) e
# minuscolo: illeggibile via desktop remoto. Si carica un TrueType vero.
_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(_FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()

# Giudizio dato guardando le sei coppie (vedi README): serve il warp solo dove
# il retoucher ha davvero raddrizzato il capo, non dove ha solo riposizionato.
CAPI = {
    "GNN10258-W-OF": "SERVE WARP — le clip tirano la vita in due picchi, nel dopo e' dritta",
    "CMC10778-M-AA": "SERVE WARP — spalla sinistra piu' bassa, nel dopo simmetrica",
    "TSH11514-W-AD": "SERVE WARP — le righe scendono verso destra, nel dopo orizzontali",
    "VST10509-W-VE": "NO WARP — corpo identico, cambiano solo le spalline",
    "FLP10848-W-86": "NO WARP — era gia' dritta",
    "BRM10465-M-AA": "DUBBIO — piu' largo nel dopo, ma puo' essere il riposizionamento",
}


def _flatten(img: Image.Image) -> Image.Image:
    """I PNG di PRIMA hanno alpha: senza appiattire su bianco il confronto
    con il JPEG del DOPO sarebbe falsato."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _scale_to_height(img: Image.Image, h: int) -> Image.Image:
    return img.resize((round(img.width * h / img.height), h), Image.LANCZOS)


def build(stem: str, nota: str) -> Path:
    prima = _scale_to_height(_flatten(Image.open(SRC / "PRIMA" / f"{stem}.png")), H)
    dopo = _scale_to_height(_flatten(Image.open(SRC / "DOPO" / f"{stem}.jpg")), H)

    w = prima.width + GAP + dopo.width
    canvas = Image.new("RGB", (w, H + LABEL_H), "white")
    canvas.paste(prima, (0, LABEL_H))
    canvas.paste(dopo, (prima.width + GAP, LABEL_H))

    d = ImageDraw.Draw(canvas)
    f_tit = _font(34, bold=True)
    f_nota = _font(28)
    d.text((10, 8), f"{stem}   —   PRIMA (scatto)", fill="#111", font=f_tit)
    d.text((prima.width + GAP + 10, 8), "DOPO (Photoshop, fatto a mano)",
           fill="#111", font=f_tit)
    # il verdetto e' in colore: verde = serve il warp, grigio = non serve
    colore = "#0a7d2e" if nota.startswith("SERVE") else "#666"
    d.text((10, 52), nota, fill=colore, font=f_nota)
    # riga di separazione: aiuta a non confondere i due capi a colpo d'occhio
    d.rectangle([prima.width, LABEL_H, prima.width + GAP, H + LABEL_H], fill="#d8d8dc")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}_confronto.jpg"
    canvas.save(path, quality=92)
    return path


def main():
    for stem, nota in CAPI.items():
        p = build(stem, nota)
        print(f"{p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
