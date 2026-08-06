"""
Deformazioni geometriche del capo — un "Liquify" vincolato.

Principio guida, ereditato dalla rimozione gruccia: **spostare i pixel che
esistono, non generarne di nuovi**. I sette motori di fill provati hanno
mostrato che i modelli che "ragionano" ricostruiscono cio' che dovrebbero
togliere; qui vale lo stesso, quindi niente generativo — solo rimappatura.

L'operazione principale e' `warp_locale()`: l'operatore indica dove e di
quanto spostare, il resto del capo segue in modo smorzato. E' quello che
serve per correzioni come "la vita di questa gonna e' storta rispetto al
resto della gonna" — dove una rotazione globale non aiuta, perche' ruoterebbe
anche cio' che e' gia' giusto.

`raddrizza()` (rotazione rigida) resta disponibile per il caso in cui e'
davvero tutto il capo a pendere, ma non e' la strada principale.
"""
import cv2
import numpy as np


def _bordo_bianco(rgb):
    """Colore con cui riempire le zone scoperte dalla deformazione.
    Le foto prodotto hanno fondo bianco uniforme, quindi il bianco non si
    nota; si legge dagli angoli per sicurezza."""
    h, w = rgb.shape[:2]
    ang = np.concatenate([
        rgb[0:10, 0:10].reshape(-1, 3), rgb[0:10, w-10:w].reshape(-1, 3),
        rgb[h-10:h, 0:10].reshape(-1, 3), rgb[h-10:h, w-10:w].reshape(-1, 3),
    ])
    return tuple(int(v) for v in np.median(ang, axis=0))


def raddrizza(rgb, angolo, alpha=None, centro=None):
    """Rotazione rigida dell'intera immagine. Non deforma nulla: le pieghe e
    le proporzioni restano identiche, cambia solo l'orientamento."""
    h, w = rgb.shape[:2]
    if abs(angolo) < 1e-3:
        return rgb.copy(), (alpha.copy() if alpha is not None else None)

    if centro is None:
        centro = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centro, angolo, 1.0)
    bordo = _bordo_bianco(rgb)

    out = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=bordo)
    out_a = None
    if alpha is not None:
        out_a = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, out_a


# --------------------------------------------------------------- warp locale

def _campo_da_trascinamenti(h, w, tratti, congelate=None, passo=8):
    """Costruisce il campo di spostamento INVERSO (per ogni pixel di uscita,
    di quanto andarlo a prendere altrove nell'immagine di partenza).

    Ogni tratto e' un trascinamento: un punto di partenza, uno di arrivo, un
    raggio d'influenza. L'effetto sfuma con una gaussiana, cosi' il tessuto
    si deforma con continuita' invece di strapparsi al bordo della zona.

    Si lavora su una griglia rada e poi si ingrandisce: il campo e' liscio,
    calcolarlo su ogni pixel di un'immagine da 4000px sarebbe lento e inutile.
    """
    gh, gw = h // passo + 2, w // passo + 2
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    gx *= passo
    gy *= passo

    dx = np.zeros((gh, gw), np.float32)
    dy = np.zeros((gh, gw), np.float32)

    for t in tratti:
        x0, y0 = float(t["x0"]), float(t["y0"])
        x1, y1 = float(t["x1"]), float(t["y1"])
        r = max(float(t.get("raggio", 200.0)), 1.0)
        forza = float(t.get("forza", 1.0))
        vx, vy = (x1 - x0) * forza, (y1 - y0) * forza
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            continue

        # peso gaussiano attorno al punto di PARTENZA del trascinamento
        d2 = (gx - x0) ** 2 + (gy - y0) ** 2
        peso = np.exp(-d2 / (2.0 * (r / 2.0) ** 2))
        peso[d2 > (r * 1.6) ** 2] = 0.0     # taglio netto oltre il raggio

        # campo inverso: il pixel di destinazione va preso indietro di v
        dx -= vx * peso
        dy -= vy * peso

    # zone congelate: l'operatore ha detto "qui non toccare". Si azzera il
    # campo dentro, con una sfumatura per non creare un gradino visibile.
    if congelate:
        blocco = np.zeros((gh, gw), np.float32)
        for z in congelate:
            zx0, zy0 = float(z["x0"]) / passo, float(z["y0"]) / passo
            zx1, zy1 = float(z["x1"]) / passo, float(z["y1"]) / passo
            xa, xb = int(min(zx0, zx1)), int(max(zx0, zx1))
            ya, yb = int(min(zy0, zy1)), int(max(zy0, zy1))
            blocco[max(ya, 0):yb + 1, max(xa, 0):xb + 1] = 1.0
        if blocco.max() > 0:
            k = max(3, (int(60 / passo) | 1))
            blocco = cv2.GaussianBlur(blocco, (k, k), 0)
            dx *= (1.0 - blocco)
            dy *= (1.0 - blocco)

    dxf = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    dyf = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)
    return dxf, dyf


def warp_locale(rgb, tratti, alpha=None, congelate=None):
    """Applica deformazioni locali indicate a mano — il "Liquify".

    `tratti`: lista di dict con x0,y0 (punto preso), x1,y1 (dove va portato),
              raggio (px, quanto tessuto attorno si trascina), forza (0-1).
              Le coordinate sono in pixel dell'immagine a piena risoluzione.
    `congelate`: rettangoli {x0,y0,x1,y1} che restano fermi (es. l'orlo,
              mentre si lavora sulla vita).

    Restituisce (rgb, alpha, info). Nessun pixel viene inventato: si
    ridistribuisce quello che c'e'.
    """
    h, w = rgb.shape[:2]
    tratti = [t for t in (tratti or [])
              if abs(t["x1"] - t["x0"]) > 0.5 or abs(t["y1"] - t["y0"]) > 0.5]
    if not tratti:
        return rgb.copy(), (alpha.copy() if alpha is not None else None), \
               {"tratti": 0, "spostamento_max": 0.0}

    dx, dy = _campo_da_trascinamenti(h, w, tratti, congelate)

    map_x = np.arange(w, dtype=np.float32)[None, :] + dx
    map_y = np.arange(h, dtype=np.float32)[:, None] + dy

    bordo = _bordo_bianco(rgb)
    out = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=bordo)
    out_a = None
    if alpha is not None:
        out_a = cv2.remap(alpha, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    spost = float(np.sqrt(dx ** 2 + dy ** 2).max())
    return out, out_a, {"tratti": len(tratti), "spostamento_max": round(spost, 1)}


def appiattisci_curva(rgb, punti, alpha=None, altezza_fascia=None,
                      congelate=None, quota=None):
    """Porta TUTTI i punti indicati alla stessa quota.

    Serve quando la linea da correggere non è una retta inclinata ma una
    **curva**: la vita di una gonna tirata dalle clip forma una V, con due
    picchi laterali e un avvallamento al centro. Con due soli punti si può
    solo ruotare la congiungente — la V resterebbe, ruotata (errore osservato
    sul capo GNN10258).

    Ogni punto diventa un'ancora che viene spinta alla quota comune, quindi
    i picchi scendono e gli avvallamenti salgono: la curva si appiattisce
    davvero.

    `punti`: [(x,y), ...] in pixel, almeno 2, tipicamente 3-5 lungo la linea.
    `quota`: y di destinazione; se None si usa la mediana (robusta rispetto
             a un punto messo male).
    """
    pts = [(float(x), float(y)) for x, y in (punti or [])]
    if len(pts) < 2:
        return rgb.copy(), (alpha.copy() if alpha is not None else None), \
               {"tratti": 0, "spostamento_max": 0.0, "nota": "servono almeno 2 punti"}

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    larghezza = max(xs) - min(xs)
    if larghezza < 5:
        return rgb.copy(), (alpha.copy() if alpha is not None else None), \
               {"tratti": 0, "spostamento_max": 0.0, "nota": "punti troppo vicini"}

    y_target = float(np.median(ys)) if quota is None else float(quota)
    # la fascia coinvolta scala con la distanza tra i punti: abbastanza da
    # far seguire il tessuto, non tanto da trascinare tutto il capo
    raggio = float(altezza_fascia) if altezza_fascia else max(
        40.0, larghezza / max(1, len(pts) - 1) * 1.3)

    tratti = [{"x0": x, "y0": y, "x1": x, "y1": y_target,
               "raggio": raggio, "forza": 1.0} for x, y in pts]

    out, out_a, info = warp_locale(rgb, tratti, alpha, congelate)
    dislivello = max(ys) - min(ys)
    info["nota"] = (f"{len(pts)} punti portati a quota {y_target:.0f} "
                    f"(dislivello corretto {dislivello:.0f}px, fascia {raggio:.0f}px)")
    return out, out_a, info


def raddrizza_linea(rgb, p0, p1, alpha=None, altezza_fascia=None,
                    congelate=None, orizzontale=True):
    """Porta in orizzontale (o verticale) una linea indicata dall'operatore.

    E' il caso tipico: "la vita di questa gonna e' storta". Si segnano due
    punti sulla vita e la si raddrizza, deformando solo la fascia attorno —
    il resto della gonna resta com'e'.

    `altezza_fascia`: quanto tessuto sopra/sotto viene coinvolto. Se None,
    si usa la lunghezza della linea, che di solito e' una scelta sensata.
    """
    (x0, y0), (x1, y1) = p0, p1
    lung = float(np.hypot(x1 - x0, y1 - y0))
    if lung < 5:
        return rgb.copy(), (alpha.copy() if alpha is not None else None), \
               {"tratti": 0, "spostamento_max": 0.0, "nota": "linea troppo corta"}

    raggio = float(altezza_fascia) if altezza_fascia else lung * 0.55

    # bersaglio: la quota media, cosi' la linea si appiattisce senza spostare
    # il capo verso l'alto o il basso
    if orizzontale:
        ym = (y0 + y1) / 2.0
        estremi = [(x0, y0, x0, ym), (x1, y1, x1, ym)]
    else:
        xm = (x0 + x1) / 2.0
        estremi = [(x0, y0, xm, y0), (x1, y1, xm, y1)]

    # punti intermedi: senza questi la fascia centrale resta indietro e la
    # linea si raddrizza solo alle estremita'
    tratti = []
    for k in np.linspace(0.0, 1.0, 7):
        px = x0 + (x1 - x0) * k
        py = y0 + (y1 - y0) * k
        if orizzontale:
            ty = (y0 + y1) / 2.0
            tratti.append({"x0": px, "y0": py, "x1": px, "y1": ty,
                           "raggio": raggio, "forza": 1.0})
        else:
            tx = (x0 + x1) / 2.0
            tratti.append({"x0": px, "y0": py, "x1": tx, "y1": py,
                           "raggio": raggio, "forza": 1.0})

    out, out_a, info = warp_locale(rgb, tratti, alpha, congelate)
    info["nota"] = (f"linea raddrizzata su {lung:.0f}px, "
                    f"fascia {raggio:.0f}px")
    return out, out_a, info
