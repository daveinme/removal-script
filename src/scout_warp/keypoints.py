"""
Stima dell'inclinazione del capo a partire dalla sua sagoma.

Perche' questo approccio e non keypoint semantici (spalle/collo/orlo stimati
da un modello): guardando i sei confronti PRIMA/DOPO fatti a mano in
Photoshop, la correzione applicata dal retoucher e' sempre la stessa cosa —
**raddrizzare l'asse orizzontale del capo** (linea spalle per i capi sopra,
linea vita per gonne e pantaloni). Non e' una deformazione libera del
tessuto. Quindi non serve stimare punti anatomici: basta misurare di quanto
il capo pende, e quella misura si legge dalla sagoma.

Casi di riferimento (in output/iterazioni/v16_warping_riferimento/):
  GNN10258  le due clip tirano la vita in due picchi -> nel dopo e' dritta
  CMC10778  spalla sinistra piu' bassa               -> nel dopo simmetrica
  TSH11514  righe che scendono verso destra          -> nel dopo orizzontali
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class Inclinazione:
    """Misura di quanto il capo pende, piu' i dati per giudicare la misura."""
    angolo: float          # gradi; >0 = il capo pende a destra (va ruotato in senso antiorario)
    y_banda: int           # riga dell'immagine su cui e' stata misurata la spalla/vita
    x_sx: int              # estremo sinistro del capo su quella banda
    x_dx: int              # estremo destro
    y_sx: float            # quota del bordo superiore a sinistra
    y_dx: float            # quota a destra
    confidenza: float      # 0-1: quanto la misura e' affidabile
    nota: str = ""

    def to_dict(self):
        return asdict(self)


def maschera_capo(rgb, alpha=None, soglia_bianco=244):
    """Sagoma del capo come maschera binaria.

    Se c'e' l'alpha (PNG StyleShoots) e' gia' la risposta della macchina e si
    usa quello. Altrimenti si separa dal fondo bianco, che nelle foto prodotto
    e' uniforme.
    """
    if alpha is not None:
        m = (alpha >= 128).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        m = (gray < soglia_bianco).astype(np.uint8) * 255

    # chiude i buchi interni (bottoni chiari, riflessi) e toglie i granelli:
    # senza questo il profilo superiore risulta seghettato e l'angolo balla
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    # tiene solo la componente piu' grande: nastro adesivo e frammenti sparsi
    # falserebbero il bounding box
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8) * 255
    return m


def corpo_capo(mask, fraz_larghezza=0.16):
    """Isola il corpo del capo scartando le appendici sottili in alto.

    Serve perche' sulle foto grezze la sagoma include ancora gruccia, gancio e
    staffa della macchina: sono attaccati al capo, quindi la componente
    connessa piu' grande li comprende, e il "profilo superiore" finisce per
    essere il gancio invece delle spalle. Misurando li' si ottengono angoli
    assurdi (visti +63° su un vestito che pende di pochi gradi).

    Criterio: si scende dall'alto e si tiene la prima riga abbastanza larga
    da essere capo e non ferramenta. Gruccia e gancio sono stretti; spalle,
    vita e maniche no.

    Sull'immagine gia' ripulita dalla gruccia questo passaggio non toglie
    nulla: la prima riga larga e' gia' la riga delle spalle.
    """
    larghezze = (mask > 0).sum(axis=1)
    if larghezze.max() == 0:
        return mask, 0
    soglia = larghezze.max() * fraz_larghezza
    righe = np.nonzero(larghezze >= soglia)[0]
    if len(righe) == 0:
        return mask, 0
    y_start = int(righe[0])
    out = mask.copy()
    out[:y_start] = 0

    # dopo il taglio la parte tenuta puo' essersi spezzata (es. le due
    # spalline del vestito): si riprende la componente piu' grande
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = (lab == big).astype(np.uint8) * 255
    return out, y_start


def _profilo_superiore(mask, x0, x1):
    """Per ogni colonna, la prima riga in cui compare il capo.
    E' il profilo delle spalle (o della vita, per una gonna appesa)."""
    sub = mask[:, x0:x1]
    ha = sub.max(axis=0) > 0
    prima = np.argmax(sub > 0, axis=0).astype(float)
    prima[~ha] = np.nan
    return prima


def stima_inclinazione(rgb, alpha=None, frazione_banda=0.18,
                       margine_laterale=0.12) -> Inclinazione:
    """Misura di quanto il capo e' storto.

    Il metodo: si prende il profilo superiore della sagoma nella fascia alta
    del capo e si adatta una retta ai due terzi laterali, saltando la parte
    centrale. Il centro va escluso perche' e' li' che stanno collo, colletto e
    scollatura — dettagli che non seguono la linea delle spalle e che
    sbilancerebbero la retta.
    """
    mask = maschera_capo(rgb, alpha)
    # scarta gruccia/gancio/staffa: senza questo si misura la ferramenta
    mask, y_tagliato = corpo_capo(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) < 500:
        return Inclinazione(0.0, 0, 0, 0, 0.0, 0.0, 0.0,
                            "sagoma troppo piccola o non trovata")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    larghezza, altezza = x1 - x0, y1 - y0
    if larghezza < 50 or altezza < 50:
        return Inclinazione(0.0, 0, 0, 0, 0.0, 0.0, 0.0, "sagoma degenere")

    prof = _profilo_superiore(mask, x0, x1)

    # fascia alta: la linea spalle/vita sta li'. Piu' in basso il profilo
    # segue le maniche o la gonna che si allarga, che non dicono l'inclinazione.
    lim_y = y0 + altezza * frazione_banda
    col_x = np.arange(x0, x1, dtype=float)
    valido = np.isfinite(prof) & (prof <= lim_y)

    # esclude il centro (collo/scollatura) e gli estremi (angoli arrotondati)
    rel = (col_x - x0) / max(larghezza, 1)
    valido &= (rel > margine_laterale) & (rel < 1 - margine_laterale)
    valido &= ~((rel > 0.38) & (rel < 0.62))

    if valido.sum() < 30:
        return Inclinazione(0.0, int(lim_y), x0, x1, 0.0, 0.0, 0.0,
                            "profilo superiore troppo frammentato per misurare")

    xv, yv = col_x[valido], prof[valido]

    # retta robusta: un solo passaggio di reiezione degli outlier a 2 sigma.
    # Serve contro etichette sporgenti e residui di gruccia non rimossi, che
    # altrimenti inclinano la retta da soli.
    coef = np.polyfit(xv, yv, 1)
    resid = yv - np.polyval(coef, xv)
    s = float(resid.std())
    if s > 1e-6:
        tieni = np.abs(resid) < 2.0 * s
        if tieni.sum() >= 20:
            xv, yv = xv[tieni], yv[tieni]
            coef = np.polyfit(xv, yv, 1)

    pendenza = float(coef[0])
    angolo = float(np.degrees(np.arctan(pendenza)))

    # confidenza: quanto i punti stanno davvero su una retta, normalizzato
    # sulla larghezza del capo. Dispersione alta = profilo irregolare
    # (drappeggi, balze) = misura da non applicare alla cieca.
    resid2 = yv - np.polyval(coef, xv)
    disp = float(np.sqrt(np.mean(resid2 ** 2)))
    conf = float(np.clip(1.0 - disp / (0.05 * altezza + 1e-6), 0.0, 1.0))
    copertura = float(valido.sum()) / max(larghezza, 1)
    conf *= float(np.clip(copertura / 0.5, 0.0, 1.0))

    y_sx = float(np.polyval(coef, x0))
    y_dx = float(np.polyval(coef, x1))

    return Inclinazione(
        angolo=angolo, y_banda=int(lim_y), x_sx=x0, x_dx=x1,
        y_sx=y_sx, y_dx=y_dx, confidenza=round(conf, 3),
        nota=(f"misurata su {int(valido.sum())} colonne, dispersione {disp:.1f}px"
              + (f", scartati {y_tagliato}px in alto (gruccia/ferramenta)"
                 if y_tagliato > 0 else "")),
    )
