"""
scout_warp — correzione della forma del capo per le foto prodotto Scout.

Modulo autonomo: nessuna dipendenza da FastAPI, dalla GUI o da scout_api.
Dipende solo da numpy e OpenCV. Lo importano allo stesso modo:

  - la GUI (gui/app.py), per il lavoro interattivo capo per capo
  - la pipeline batch (scout_api/routes/pipeline.py), dopo la rimozione
    gruccia e prima di normalize/positioning

## Il modello di lavoro: semi-manuale

L'operatore indica **dove** e **quanto** correggere; il modulo esegue la
deformazione. Non c'e' nessuna stima automatica nella strada principale, per
una ragione verificata sui capi di riferimento: il profilo di un capo scende
in diagonale per costruzione (le spalle sono inclinate di loro), quindi
misurarne l'inclinazione non dice se il capo e' storto. Un tentativo in quel
senso ha prodotto angoli privi di senso (+63° su un vestito che pende di 2°).

Il caso tipico che questo modulo risolve: *"la vita di questa gonna e' storta
rispetto al resto della gonna"*. Una rotazione globale non serve — ruoterebbe
anche cio' che e' gia' giusto. Serve deformare una zona lasciando ferma
l'altra, come il Liquify di Photoshop.

Uso:

    from scout_warp import applica_tratti, applica_linea

    # trascinamenti liberi (x0,y0 -> x1,y1, con raggio d'influenza)
    out, out_a, info = applica_tratti(rgb, tratti, alpha)

    # oppure: raddrizza una linea indicata con due punti
    out, out_a, info = applica_linea(rgb, (x0,y0), (x1,y1), alpha)
"""
from .keypoints import (Inclinazione, corpo_capo, maschera_capo,
                        stima_inclinazione)
from .transform import (appiattisci_curva, raddrizza, raddrizza_linea,
                        warp_locale)

__all__ = [
    "applica_tratti", "applica_linea", "applica_curva", "applica_rotazione",
    "warp_locale", "raddrizza_linea", "appiattisci_curva", "raddrizza",
    "maschera_capo", "corpo_capo",
    # la stima automatica resta esposta ma NON e' la strada principale:
    # vedi la nota sopra sul perche' non e' affidabile
    "stima_inclinazione", "Inclinazione",
]


def applica_tratti(rgb, tratti, alpha=None, congelate=None):
    """Deformazioni locali indicate a mano — il caso d'uso principale.

    `tratti`: [{x0,y0,x1,y1,raggio,forza}, ...] in pixel dell'immagine piena.
              Ogni tratto prende il tessuto in (x0,y0) e lo porta in (x1,y1);
              l'intorno segue sfumando entro `raggio`.
    `congelate`: rettangoli {x0,y0,x1,y1} che restano fermi.
    """
    return warp_locale(rgb, tratti, alpha, congelate)


def applica_linea(rgb, p0, p1, alpha=None, altezza_fascia=None,
                  congelate=None, orizzontale=True):
    """Raddrizza una linea del capo (vita, orlo, spalle) indicata con due
    punti, deformando solo la fascia attorno."""
    return raddrizza_linea(rgb, p0, p1, alpha,
                           altezza_fascia=altezza_fascia,
                           congelate=congelate, orizzontale=orizzontale)


def applica_curva(rgb, punti, alpha=None, altezza_fascia=None,
                  congelate=None, quota=None):
    """Appiattisce una linea CURVA portando tutti i punti alla stessa quota.

    È il caso reale più frequente: la vita di una gonna tirata dalle clip non
    è una retta inclinata ma una V. Con 2 soli punti si ruota la
    congiungente e la V resta; con 3-5 punti lungo la curva, i picchi
    scendono e il centro sale.
    """
    return appiattisci_curva(rgb, punti, alpha, altezza_fascia=altezza_fascia,
                             congelate=congelate, quota=quota)


def applica_rotazione(rgb, angolo, alpha=None):
    """Rotazione rigida dell'intero capo.

    Serve solo quando e' davvero tutto il capo a pendere. Nella maggior parte
    dei casi reali la correzione utile e' locale, non globale: questa
    funzione resta per completezza.
    """
    import numpy as np

    out_a = None
    if abs(angolo) < 1e-3:
        return rgb.copy(), (alpha.copy() if alpha is not None else None), \
               {"angolo": 0.0, "nota": "nessuna modifica"}

    # centro del capo, non del canvas: ruotare attorno al centro
    # dell'immagine sposterebbe lateralmente un capo non centrato
    m, _ = corpo_capo(maschera_capo(rgb, alpha))
    ys, xs = np.nonzero(m)
    centro = ((float(xs.min()) + float(xs.max())) / 2.0,
              (float(ys.min()) + float(ys.max())) / 2.0) if len(xs) else None
    out, out_a = raddrizza(rgb, angolo, alpha, centro=centro)
    return out, out_a, {"angolo": round(float(angolo), 2),
                        "nota": f"rotazione {angolo:+.2f}°"}
