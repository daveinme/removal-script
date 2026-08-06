"""
Correzione guidata dalla fisica: il campo di spostamento si COSTRUISCE dai
punti in cui le clip tirano il capo, invece di cercarlo nel gradiente.

Il ragionamento (dal consulto in `pareri/3_DeepSeek...`):

  Il tessuto è tirato su in due punti dalle clip; il resto (balze, orlo)
  pende dritto. La FORMA del campo di spostamento è quindi determinata dalla
  meccanica — una membrana tirata in due nodi, che decade a zero verso la
  zona non deformata. L'unica incognita vera è **quanto** tira ciascuna clip:
  due scalari s_L, s_R. Non una linea, non un angolo.

Perché è meglio del rilevamento per gradiente (fallito due volte):

- **Texture-agnostic**: funziona su tessuto uniforme e a quadri, cioè i capi
  che il rilevatore per gradiente doveva rifiutare ("nessuna linea netta").
- **2 soli gradi di libertà**: difficile produrre risultati assurdi come il
  +47.3° che nasceva dall'agganciare due balze diverse.
- **Nessun problema di corrispondenza sinistra/destra**: non si cercano due
  picchi indipendenti, quindi non possono riferirsi a strutture diverse.
- Se la stima automatica non convince, l'operatore regola due numeri con uno
  slider — molto più semplice che tracciare una linea a mano.

Coerente col vincolo del progetto: si ridistribuiscono i pixel esistenti,
non se ne generano.
"""
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

from .keypoints import corpo_capo, maschera_capo


@dataclass
class Presa:
    """Un punto in cui una clip afferra il capo."""
    x: int
    y: int
    lato: str          # "sx" | "dx"
    larghezza: int     # ampiezza del contatto, usata per il raggio del kernel

    def to_dict(self):
        return asdict(self)


@dataclass
class Diagnosi:
    prese: list = field(default_factory=list)
    s_sx: float = 0.0          # px di tiro stimato sulla clip sinistra
    s_dx: float = 0.0
    asse_gradi: float = 0.0    # inclinazione dell'asse di caduta del capo
    y_riferimento: float = 0.0
    ok: bool = False
    motivo: str = ""

    def to_dict(self):
        d = asdict(self)
        d["prese"] = [p.to_dict() if isinstance(p, Presa) else p for p in self.prese]
        return d


# --------------------------------------------------------------- punti presa

def punti_presa(mask_gruccia, mask_capo, min_area_frac=0.002):
    """I punti in cui le clip toccano il capo.

    Sono gratis: la maschera della gruccia è già stata calcolata da SAM 3 per
    l'inpainting. I punti di presa sono i **minimi in y** (cioè il punto più
    basso) di ogni componente della maschera gruccia che entra nel capo.

    NB: va usata la maschera PRIMA dell'inpainting. Dopo LaMa quell'evidenza
    non esiste più — è l'errore che ha reso incoerenti le misure precedenti.
    """
    h, w = mask_capo.shape[:2]
    inter = ((mask_gruccia > 0) & (mask_capo > 0)).astype(np.uint8)
    if inter.sum() == 0:
        return []

    n, lab, stats, _ = cv2.connectedComponentsWithStats(inter, 8)
    area_min = max(50, int(min_area_frac * h * w * 0.01))
    prese = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < area_min:
            continue
        ys, xs = np.nonzero(lab == i)
        # il punto di presa è il più BASSO del contatto: è lì che il tessuto
        # viene effettivamente strozzato
        y_max = int(ys.max())
        x_c = int(np.median(xs[ys > y_max - max(3, (ys.max() - ys.min()) // 10)]))
        prese.append(Presa(x=x_c, y=y_max, lato="",
                           larghezza=int(stats[i, cv2.CC_STAT_WIDTH])))

    if not prese:
        return []
    prese.sort(key=lambda p: p.x)
    # etichetta sinistra/destra rispetto al centro del capo
    xs_capo = np.nonzero(mask_capo.any(axis=0))[0]
    cx = (int(xs_capo.min()) + int(xs_capo.max())) / 2.0 if len(xs_capo) else w / 2
    for p in prese:
        p.lato = "sx" if p.x < cx else "dx"
    return prese


# ------------------------------------------------------------ asse di caduta

def asse_di_caduta(mask_capo, dal=0.55, al=0.95):
    """Inclinazione dell'asse verticale del capo, misurata nella zona BASSA.

    "Dritto" non significa "orizzontale": va misurato dalla parte che le clip
    non hanno deformato. Definirlo dalla vita stessa sarebbe circolare — è
    l'errore del tentativo precedente.

    Restituisce i gradi di inclinazione dell'asse (0 = perfettamente
    verticale).
    """
    ys = np.nonzero(mask_capo.any(axis=1))[0]
    if len(ys) < 20:
        return 0.0
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0

    centri = []
    for y in range(int(y0 + h * dal), int(y0 + h * al), max(1, h // 120)):
        xs = np.nonzero(mask_capo[y])[0]
        if len(xs) > 10:
            centri.append((float(xs.mean()), float(y)))
    if len(centri) < 8:
        return 0.0

    c = np.array(centri)
    # retta x = a*y + b: la pendenza dice quanto l'asse pende
    a = np.polyfit(c[:, 1], c[:, 0], 1)[0]
    return float(np.degrees(np.arctan(a)))


# ------------------------------------------------------------ stima del tiro

def stima_tiro(rgb, mask_capo, prese, asse_gradi=0.0, banda_frac=0.06):
    """Stima quanto ciascuna clip ha tirato su il tessuto.

    Metodo: all'altezza di ogni presa si guarda il bordo superiore del capo
    nelle immediate vicinanze, e lo si confronta con la quota di riferimento
    che quel bordo avrebbe **senza** il tiro — stimata dal tessuto laterale
    non strozzato dalla clip.

    Volutamente semplice: due scalari misurati su un intorno ampio sono molto
    più robusti di una linea intera stimata su un gradiente ambiguo.
    """
    h, w = mask_capo.shape[:2]
    ys_capo = np.nonzero(mask_capo.any(axis=1))[0]
    if len(ys_capo) == 0 or not prese:
        return {}, 0.0

    altezza = int(ys_capo.max() - ys_capo.min())
    banda = max(20, int(altezza * banda_frac))

    def bordo_sup(x0, x1):
        """quota media del bordo superiore del capo tra due colonne"""
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        if x1 - x0 < 3:
            return None
        q = []
        for x in range(x0, x1):
            col = np.nonzero(mask_capo[:, x])[0]
            if len(col):
                q.append(float(col[0]))
        return float(np.median(q)) if len(q) > 3 else None

    tiri = {}
    quote_sane = []
    for p in prese:
        lar = max(p.larghezza, int(w * 0.02))
        # quota SOTTO la clip (tessuto strozzato)
        y_clip = bordo_sup(p.x - lar * 0.4, p.x + lar * 0.4)
        # quota LATERALE (tessuto sano, non tirato): si guarda al di là della
        # clip, dalla parte verso l'esterno del capo
        off = int(lar * 1.2)
        if p.lato == "sx":
            y_sano = bordo_sup(p.x + off, p.x + off + lar)
        else:
            y_sano = bordo_sup(p.x - off - lar, p.x - off)
        if y_clip is None or y_sano is None:
            continue
        # tiro = di quanto il tessuto sotto la clip sta PIÙ IN ALTO del sano
        tiri[p.lato] = float(y_sano - y_clip)
        quote_sane.append(y_sano)

    y_rif = float(np.median(quote_sane)) if quote_sane else 0.0
    return tiri, y_rif


# ------------------------------------------------------------------- analisi

def analizza(rgb, mask_gruccia, alpha=None, max_tiro_frac=0.10):
    """Misura la deformazione causata dalle clip. Non modifica l'immagine.

    `mask_gruccia` è la maschera SAM 3 usata per l'inpainting, e `rgb`
    dovrebbe essere l'immagine PRIMA del fill: è lì che l'evidenza del tiro
    esiste ancora.
    """
    mask_capo, _ = corpo_capo(maschera_capo(rgb, alpha))
    prese = punti_presa(mask_gruccia, mask_capo)
    if len(prese) < 1:
        return Diagnosi(motivo="nessun punto di presa trovato: la maschera "
                               "della gruccia non tocca il capo")

    asse = asse_di_caduta(mask_capo)
    tiri, y_rif = stima_tiro(rgb, mask_capo, prese, asse)
    if not tiri:
        return Diagnosi(prese=prese, asse_gradi=asse,
                        motivo="impossibile misurare il tiro accanto alle clip")

    ys = np.nonzero(mask_capo.any(axis=1))[0]
    altezza = int(ys.max() - ys.min())
    limite = max_tiro_frac * altezza

    s_sx = float(np.clip(tiri.get("sx", 0.0), -limite, limite))
    s_dx = float(np.clip(tiri.get("dx", 0.0), -limite, limite))

    d = Diagnosi(prese=prese, s_sx=round(s_sx, 1), s_dx=round(s_dx, 1),
                 asse_gradi=round(asse, 2), y_riferimento=round(y_rif, 1))
    if abs(s_sx) < 3 and abs(s_dx) < 3:
        d.motivo = (f"tiro trascurabile (sx {s_sx:+.0f}px, dx {s_dx:+.0f}px): "
                    f"il capo è già a posto")
        d.ok = False
    else:
        d.ok = True
        d.motivo = (f"tiro stimato — sinistra {s_sx:+.0f}px, destra "
                    f"{s_dx:+.0f}px; asse di caduta {asse:+.1f}°")
    return d


# ------------------------------------------------------------------- correzione

def campo_da_prese(shape, prese, s_sx, s_dx, y_congela, raggio_x, raggio_y):
    """Costruisce il campo di spostamento verticale dalle prese.

    `u(x,y) = Σ_i s_i · K(x - x_i, y - y_i)`, con K gaussiano anisotropo e
    spostamento forzato a zero da `y_congela` in giù, così la parte bassa del
    capo (che non è deformata) resta immobile.
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = np.zeros((h, w), np.float32)

    for p in prese:
        s = s_sx if p.lato == "sx" else s_dx
        if abs(s) < 1e-3:
            continue
        k = np.exp(-(((xx - p.x) ** 2) / (2 * raggio_x ** 2) +
                     ((yy - p.y) ** 2) / (2 * raggio_y ** 2)))
        # il tessuto è stato tirato SU: per rimettere a posto va spinto GIÙ
        u += np.float32(s) * k.astype(np.float32)

    # smorzamento verso il basso: a y_congela il campo è già nullo
    if y_congela and y_congela < h:
        ramp = np.clip((y_congela - yy) / max(1.0, y_congela * 0.35), 0.0, 1.0)
        u *= ramp.astype(np.float32)
    return u


def applica(rgb, diagnosi, alpha=None, s_sx=None, s_dx=None,
            forza=1.0, y_congela_frac=0.62):
    """Applica la correzione fisica. `s_sx`/`s_dx` sovrascrivono la stima
    automatica: è il gancio per lo slider dell'operatore."""
    mask_capo, _ = corpo_capo(maschera_capo(rgb, alpha))
    ys = np.nonzero(mask_capo.any(axis=1))[0]
    xs = np.nonzero(mask_capo.any(axis=0))[0]
    if len(ys) == 0:
        return rgb.copy(), (alpha.copy() if alpha is not None else None), \
               {"nota": "sagoma non trovata", "s_sx": 0, "s_dx": 0}

    h, w = rgb.shape[:2]
    altezza = int(ys.max() - ys.min())
    larghezza = int(xs.max() - xs.min())

    ssx = (diagnosi.s_sx if s_sx is None else float(s_sx)) * forza
    sdx = (diagnosi.s_dx if s_dx is None else float(s_dx)) * forza

    u = campo_da_prese(
        rgb.shape, diagnosi.prese, ssx, sdx,
        y_congela=int(ys.min() + altezza * y_congela_frac),
        raggio_x=max(40.0, larghezza * 0.30),
        raggio_y=max(40.0, altezza * 0.16),
    )

    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.arange(h, dtype=np.float32)[:, None] - u

    from .transform import _bordo_bianco
    bordo = _bordo_bianco(rgb)
    out = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=bordo)
    out_a = None
    if alpha is not None:
        out_a = cv2.remap(alpha, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return out, out_a, {
        "s_sx": round(ssx, 1), "s_dx": round(sdx, 1),
        "spostamento_max": round(float(np.abs(u).max()), 1),
        "prese": len(diagnosi.prese),
        "nota": f"correzione fisica: tiro sx {ssx:+.0f}px, dx {sdx:+.0f}px",
    }
