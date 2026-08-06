"""
Rilevamento automatico della linea da raddrizzare.

Perche' questo e' diverso dal tentativo precedente (fallito): li' misuravo il
**profilo esterno** della sagoma, che scende in diagonale per costruzione —
le spalle di una camicia sono inclinate di loro, quindi quella pendenza non
dice se il capo e' storto. Qui invece si cerca una **linea interna** al capo:
il bordo della fascia in vita, la cucitura sotto il giromanica, l'orlo. Sono
elementi che nel capo reale sono orizzontali, quindi la loro inclinazione
nella foto e' esattamente il difetto da correggere.

Metodo: gradiente verticale (le linee orizzontali del tessuto danno un
segnale forte), proiettato per righe dentro la fascia alta del capo. La riga
con la risposta piu' forte e' il bordo piu' marcato; su quella si misura
l'inclinazione confrontando la meta' sinistra con la meta' destra.
"""
import cv2
import numpy as np

from .keypoints import corpo_capo, maschera_capo


def _fascia_di_ricerca(mask, dal=0.04, al=0.35):
    """Dove cercare la linea: la parte alta del capo, saltando il bordo
    superiore vero e proprio (che e' il contorno, non una linea interna)."""
    ys = np.nonzero(mask.any(axis=1))[0]
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    h = y1 - y0
    return int(y0 + h * dal), int(y0 + h * al)


def linea_vita(rgb, alpha=None, dal=0.04, al=0.35):
    """Cerca la linea orizzontale piu' marcata nella fascia alta del capo.

    Restituisce (p0, p1, diagnostica) con p0/p1 in pixel, oppure
    (None, None, diagnostica) se non trova niente di affidabile.
    """
    mask, _ = corpo_capo(maschera_capo(rgb, alpha))
    banda = _fascia_di_ricerca(mask, dal, al)
    if banda is None:
        return None, None, {"ok": False, "motivo": "sagoma non trovata"}
    ya, yb = banda

    xs = np.nonzero(mask.any(axis=0))[0]
    if len(xs) < 50:
        return None, None, {"ok": False, "motivo": "capo troppo stretto"}
    x0, x1 = int(xs.min()), int(xs.max())
    larg = x1 - x0

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), max(1.0, larg / 400.0))
    # gradiente verticale: risponde ai bordi ORIZZONTALI (cuciture, fasce)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    gy = np.abs(gy)
    gy[mask == 0] = 0.0

    # si misura separatamente su meta' sinistra e meta' destra: se la linea e'
    # storta, il picco cade a quote diverse, e la differenza E' l'inclinazione
    xm = (x0 + x1) // 2
    margine = int(larg * 0.12)
    sx = gy[ya:yb, x0 + margine:xm]
    dx = gy[ya:yb, xm:x1 - margine]
    if sx.size == 0 or dx.size == 0:
        return None, None, {"ok": False, "motivo": "fascia di ricerca vuota"}

    prof_sx = sx.mean(axis=1)
    prof_dx = dx.mean(axis=1)
    # leggero smorzamento: evita di agganciarsi a una singola riga rumorosa
    k = max(3, int((yb - ya) * 0.02) | 1)
    prof_sx = cv2.GaussianBlur(prof_sx.reshape(-1, 1), (1, k), 0).ravel()
    prof_dx = cv2.GaussianBlur(prof_dx.reshape(-1, 1), (1, k), 0).ravel()

    i_sx = int(np.argmax(prof_sx))
    i_dx = int(np.argmax(prof_dx))
    y_sx, y_dx = ya + i_sx, ya + i_dx

    # affidabilita': quanto il picco spicca sul fondo. Un tessuto uniforme
    # non ha linee, e in quel caso il massimo e' rumore.
    def nitidezza(p, i):
        pic = float(p[i])
        base = float(np.median(p))
        return (pic - base) / (pic + 1e-6)

    n_sx, n_dx = nitidezza(prof_sx, i_sx), nitidezza(prof_dx, i_dx)
    conf = float(min(n_sx, n_dx))

    px_sx = x0 + margine + (xm - x0 - margine) // 2
    px_dx = xm + (x1 - margine - xm) // 2
    dislivello = y_dx - y_sx
    ang = float(np.degrees(np.arctan2(dislivello, max(px_dx - px_sx, 1))))

    diag = {
        "ok": conf >= 0.30 and abs(ang) <= 12.0,
        "confidenza": round(conf, 3),
        "angolo": round(ang, 2),
        "dislivello_px": int(dislivello),
        "y_sx": int(y_sx), "y_dx": int(y_dx),
        "motivo": "",
    }
    if conf < 0.30:
        diag["motivo"] = (f"nessuna linea netta nella fascia alta "
                          f"(nitidezza {conf:.2f}) — tessuto uniforme?")
    elif abs(ang) > 12.0:
        diag["motivo"] = (f"inclinazione implausibile ({ang:+.1f}°): "
                          f"probabile aggancio a due elementi diversi")
    else:
        diag["motivo"] = (f"linea trovata, dislivello {dislivello:+d}px "
                          f"({ang:+.2f}°)")

    return (px_sx, y_sx), (px_dx, y_dx), diag
