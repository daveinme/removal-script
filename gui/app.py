#!/usr/bin/env python3
"""
GUI di controllo per la rimozione gruccia — Scout.

Serve a lanciare l'elaborazione capo per capo tenendo il controllo dei
parametri e vedendo subito il risultato, invece di far girare un batch alla
cieca e giudicarlo dopo.

Avvio:  python3 gui/app.py     ->  http://127.0.0.1:8095

Diverso dal sito in sito-review/ (che e' una galleria passiva dei risultati
gia' prodotti): qui si sceglie l'immagine, si regolano i parametri, si lancia
e si confronta prima/dopo affiancati.
"""
import base64
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
# src/ contiene scout_warp, il modulo di raddrizzamento: e' autonomo (non
# importa nulla della GUI) proprio perche' lo stesso codice andra' usato da
# scout_api nella pipeline batch.
sys.path.insert(0, str(ROOT / "src"))

# Le immagini caricate vivono SOLO in memoria, per la sessione corrente: niente
# scritture su disco, così un refresh riparte sempre pulito. Scelta voluta in
# fase di prototipo — quando la GUI verrà integrata in scout_api servirà una
# gestione persistente vera.
UPLOADS: dict[str, bytes] = {}

app = FastAPI(title="Controllo rimozione gruccia")

# Stato del job in corso: la UI lo interroga in polling per mostrare i log
# mentre l'elaborazione gira, invece di restare bloccata sulla richiesta.
JOB = {"running": False, "log": [], "result": None, "error": None}
_lock = threading.Lock()

# PNG a piena risoluzione dell'ultimo risultato: tenuto fuori da JOB["result"]
# cosi' /api/status resta leggero anche in polling continuo durante il job.
DOWNLOAD = {"png": None, "name": None}

# Sessione di lavoro iterativo su un capo: si parte dall'immagine originale e
# a ogni passata si accumula il risultato, cosi' la successiva lavora su un
# residuo piu' piccolo (e quindi piu' facile da riempire).
# Ogni passata viene registrata: il registro dice DOVE la parte automatica
# sbaglia sistematicamente, ed e' quello che serve per migliorarla.
SESSION = {"filename": None, "current": None, "passes": []}
SESSION_LOG = ROOT / "output" / "sessioni.jsonl"


def log(msg: str):
    with _lock:
        JOB["log"].append(msg)
    print(msg, flush=True)


def load_rgb(raw: bytes):
    """Decodifica l'immagine caricata. Se ha alpha (es. PNG StyleShoots) lo
    appiattisce su bianco, restituendo anche l'alpha originale: serve dopo per
    sapere cosa era davvero sfondo secondo la macchina."""
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError("Immagine non leggibile o formato non supportato")
    alpha = None
    if im.ndim == 3 and im.shape[2] == 4:
        alpha = im[:, :, 3].copy()
        rgb = im[:, :, :3].copy()
        rgb[alpha < 128] = 255
    else:
        rgb = im[:, :, :3].copy() if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return rgb, alpha


def to_data_uri(img_bgr, max_side=1400, quality=82):
    h, w = img_bgr.shape[:2]
    sc = min(1.0, max_side / max(h, w))
    if sc < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def banner_parametri(img_bgr, rec):
    """Disegna in alto una fascia con i parametri della passata (dilatazione,
    alpha, ombre...): serve per confrontare a colpo d'occhio piu' salvataggi
    fatti cambiando un solo parametro alla volta, senza dover riaprire il
    registro sessioni.jsonl per ricordare cosa si era usato."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    campi = [
        ("engine", "motore"), ("dilate_px", "dilate"), ("use_alpha", "alpha"),
        ("catch_shadows", "ombre"), ("shadow_strength", "soglia_ombra"),
        ("conf", "conf"),
    ]
    parti = [f"{etichetta}={rec.get(chiave)}" for chiave, etichetta in campi
             if rec.get(chiave) is not None]
    testo = "  |  ".join(parti)

    h, w = img_bgr.shape[:2]
    banner_h = max(50, h // 28)
    canvas = np.full((h + banner_h, w, 3), 255, dtype=np.uint8)
    canvas[banner_h:, :] = img_bgr

    pil_img = PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            size=max(14, banner_h // 3))
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, w, banner_h], fill=(20, 20, 20))
    draw.text((14, banner_h // 4), testo, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- detection

def free_vram(*objs):
    """Scarica dei modelli dalla GPU. Necessario perche' SAM 3 (~9 GB) e il
    fill generativo non convivono in 12 GB di VRAM: vanno usati in sequenza,
    liberando tra l'uno e l'altro."""
    import gc
    import torch
    for o in objs:
        try:
            if hasattr(o, "model") and hasattr(o.model, "to"):
                o.model.to("cpu")
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def detect_sam3(rgb, prompts, conf):
    """SAM 3 con prompt testuale. Ritorna maschera uint8 o None se non
    disponibile/non trova nulla."""
    from ultralytics.models.sam import SAM3SemanticPredictor

    # se un motore di fill precedente (ZITS/PowerPaint) e' rimasto in VRAM,
    # SAM3 (~9 GB) puo' andare in OOM: si libera prima di caricarlo.
    free_vram()

    ckpt = ROOT / "modelli" / "sam3" / "sam3.pt"
    if not ckpt.exists() or ckpt.stat().st_size < 1_000_000:
        raise FileNotFoundError(
            "Pesi SAM 3 non presenti in modelli/sam3/sam3.pt — "
            "accesso a huggingface.co/facebook/sam3 non ancora approvato."
        )
    pred = SAM3SemanticPredictor(overrides={
        "conf": conf, "task": "segment", "mode": "predict",
        "model": str(ckpt), "save": False, "verbose": False,
    })
    pred.set_image(rgb)  # array in memoria, nessun file temporaneo
    res = pred(text=prompts)[0]
    if res.masks is None or len(res.masks.data) == 0:
        free_vram(pred)
        return None, []
    m = res.masks.data.cpu().numpy()
    comb = (m.max(axis=0) > 0.5).astype(np.uint8) * 255
    if comb.shape[:2] != rgb.shape[:2]:
        comb = cv2.resize(comb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    confs = [round(float(c), 3) for c in res.boxes.conf] if res.boxes is not None else []
    # SAM 3 occupa ~9 GB: va scaricato dalla VRAM prima del fill, altrimenti
    # su una 3060 12GB il generativo non ci sta piu' e va in out-of-memory.
    free_vram(pred, res)
    return comb, confs




# ------------------------------------------------------------------- fill

def add_shadows(rgb, md, grow_px=120, strength=0.82):
    """Estende la maschera alle OMBRE che la gruccia proietta sul tessuto.

    SAM 3 segmenta l'oggetto fisico (barra, clip, gancio) ma non l'ombra che
    quell'oggetto getta sul capo sottostante. Risultato: la clip sparisce e
    resta il suo alone scuro sul tessuto — il difetto principale segnalato
    sui capi a fantasia.

    Metodo: si guarda solo l'anello attorno alla maschera gia' trovata, si
    stima la luminanza del tessuto sano li' intorno e si marcano come ombra
    i pixel sensibilmente piu' scuri. Restando ancorati alla maschera non si
    rischia di prendere le parti scure del pattern (le caffettiere grigie),
    che stanno ovunque nel capo e non solo attorno alla gruccia.
    """
    md_bool = md > 0
    if not md_bool.any():
        return md, 0

    # anello di ricerca attorno alla gruccia
    ring = cv2.dilate(md, np.ones((grow_px, grow_px), np.uint8)) > 0
    ring &= ~md_bool
    if not ring.any():
        return md, 0

    L = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    # riferimento: tessuto sano appena FUORI dall'anello, stessa zona
    outer = cv2.dilate(md, np.ones((grow_px * 2, grow_px * 2), np.uint8)) > 0
    outer &= ~ring & ~md_bool & (rgb.min(axis=2) < 235)   # esclude lo sfondo
    if outer.sum() < 500:
        return md, 0
    ref = float(np.median(L[outer]))

    # sono ombra i pixel dell'anello sensibilmente piu' scuri del tessuto sano,
    # escludendo lo sfondo bianco (li' non c'e' nulla da recuperare)
    is_fabric = rgb.min(axis=2) < 235
    shadow = ring & is_fabric & (L < ref * strength)
    if not shadow.any():
        return md, 0

    # tiene solo le macchie attaccate alla maschera: un'ombra e' contigua
    # all'oggetto che la proietta, le zone scure del pattern no
    lab_all = (shadow | md_bool).astype(np.uint8)
    n, lbl = cv2.connectedComponents(lab_all, 8)
    keep = np.unique(lbl[md_bool])
    attached = np.isin(lbl, keep[keep > 0]) & shadow

    out = md.copy()
    out[attached] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return out, int(attached.sum())


def clean_isolated(rgb, alpha, max_frac=0.01):
    """Rimuove i frammenti isolati dallo sfondo (nastro adesivo, spilli,
    puntini): componenti dell'alpha staccate dal capo e molto piccole.

    Non e' una soglia tarata sull'immagine: il capo e' sempre la componente
    dominante e i residui sono ordini di grandezza piu' piccoli (misurato:
    capo 4.732.722 px contro nastrini da ~24.000, cioe' 0,5%). Si tiene la
    componente piu' grande e si scarta il resto sotto max_frac.
    """
    if alpha is not None:
        op = (alpha >= 128).astype(np.uint8)
    else:
        # senza alpha (immagini caricate a mano): il soggetto e' cio' che non
        # e' bianco. Funziona su sfondo studio, che e' il nostro caso.
        op = (rgb.min(axis=2) < 240).astype(np.uint8)
        if op.sum() == 0:
            return rgb, 0, []
    n, lab, st, _ = cv2.connectedComponentsWithStats(op, 8)
    if n <= 2:
        return rgb, 0, []
    areas = st[1:, 4]
    main = int(np.argmax(areas)) + 1
    limit = areas.max() * max_frac
    out = rgb.copy()
    removed, sizes = 0, []
    for i in range(1, n):
        if i == main or st[i, 4] > limit:
            continue
        out[lab == i] = 255
        removed += 1
        sizes.append(int(st[i, 4]))
    return out, removed, sorted(sizes, reverse=True)[:6]


def patch_fill(rgb, md, search_px=None, patch_margin=40, min_tessuto=0.97):
    """Texture patching: riempie il buco copiando il pattern VERO dallo stesso
    capo, invece di farlo inventare a un modello.

    Per ogni componente della maschera cerca, nell'intorno, una zona di
    tessuto valida (fuori maschera) delle stesse dimensioni e la incolla con
    cv2.seamlessClone (Poisson blending), che raccorda luminosita' e colore
    ai bordi. Il pattern risultante e' quello reale del capo — nessuna
    invenzione, che e' il limite di tutti i generativi provati.

    `min_tessuto`: frazione minima di pixel non-bianchi che una finestra
    sorgente deve avere per essere accettata. Vicino a un bordo curvo o
    stretto (es. un colletto), NESSUNA traslazione della stessa forma puo'
    restare interamente dentro il tessuto: con la soglia di default (0.97)
    la ricerca fallisce sempre e la componente resta intatta, senza fill e
    senza errore — va abbassata per quei casi (rischio: puo' clonare un
    po' di sfondo/bordo dentro la maschera).

    Ritorna l'immagine e il numero di patch applicate.
    """
    out = rgb.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats((md > 0).astype(np.uint8), 8)
    applied = 0
    H, W = md.shape
    # la distanza di ricerca va scalata sull'immagine: su una foto da 5000px
    # il buco della gruccia e' alto ~500px, quindi un raggio fisso di 600px
    # non basta nemmeno a uscire dal buco stesso.
    if search_px is None:
        search_px = max(H, W) // 2

    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 200:
            continue
        # riquadro del buco, con un po' di margine per il blending
        bx0, by0 = max(0, x - patch_margin), max(0, y - patch_margin)
        bx1, by1 = min(W, x + w + patch_margin), min(H, y + h + patch_margin)
        bw, bh = bx1 - bx0, by1 - by0
        if bw < 8 or bh < 8:
            continue

        # cerca una finestra sorgente vicina, interamente fuori dalla maschera
        # e dentro il capo: si prova sopra/sotto/sinistra/destra a distanze
        # crescenti, prendendo la prima che sia "tessuto pulito"
        best = None
        step = max(24, bh // 4)
        for dist in range(max(step, int(bh * 1.1)), search_px, step):
            # si prova sotto per primo: su un capo appeso il tessuto continua
            # verso il basso, mentre sopra c'e' quasi sempre lo sfondo
            for dy, dx in ((dist, 0), (0, dist), (0, -dist), (-dist, 0),
                           (dist, dist), (dist, -dist)):
                sy0, sx0 = by0 + dy, bx0 + dx
                if sy0 < 0 or sx0 < 0 or sy0 + bh > H or sx0 + bw > W:
                    continue
                win_mask = md[sy0:sy0 + bh, sx0:sx0 + bw]
                if win_mask.any():
                    continue  # la sorgente non deve contenere altra maschera
                win = rgb[sy0:sy0 + bh, sx0:sx0 + bw]
                # deve essere quasi tutto tessuto, non sfondo bianco
                if (win.min(axis=2) < 235).mean() < min_tessuto:
                    continue
                best = (sy0, sx0)
                break
            if best:
                break
        if not best:
            continue

        sy0, sx0 = best
        src = rgb[sy0:sy0 + bh, sx0:sx0 + bw]
        patch_mask = (md[by0:by1, bx0:bx1] > 0).astype(np.uint8) * 255
        if patch_mask.sum() == 0:
            continue
        center = (bx0 + bw // 2, by0 + bh // 2)
        try:
            out = cv2.seamlessClone(src, out, patch_mask, center, cv2.NORMAL_CLONE)
            applied += 1
        except cv2.error:
            continue
    return out, applied


_IOPAINT = {}

# Motori di rimozione forniti da IOPaint. Ridotto ai soli motori con
# possibilita' reale di restare in uso (vedi consulto in pareri/): gli altri
# (MAT, MIGAN, FcF, ZITS-iopaint, LDM, LaMa-iopaint) erano solo termini di
# paragone, mai indicati come promettenti, rimossi per non appesantire la
# selezione motore.
IOPAINT_ENGINES = {
    # PowerPaint e' l'unico addestrato SPECIFICAMENTE per l'object removal:
    # ha un task dedicato ("object-remove") invece di un prompt generico.
    # E' un diffusion, ma guidato a cancellare e non a generare — quindi in
    # teoria non dovrebbe ricadere nel difetto di SD 1.5 (che inventava
    # una gruccia al posto di quella da togliere).
    "iop_powerpaint": "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
}


def inpaint_qwen(rgb, md, steps=30, quant="Q5_K_M", garment="felpa"):
    """Qwen Image Inpaint (GGUF), motore esplorativo per le occlusioni grandi
    su zone semantiche (colletto, spalle) dove LaMa appiattisce.

    Gira in venv-qwen, non nel venv della GUI: diffusers/transformers li'
    dentro sono troppo vecchi (vincolati da IOPaint) per QwenImageTransformer
    2DModel e GGUFQuantizationConfig. Si lancia quindi come subprocess con
    l'interprete di venv-qwen, passando originale+maschera come file
    temporanei — stesso principio di isolamento gia' usato per ZITS (repo
    clonato a parte) e previsto per FLUX.2 Klein (vedi requirements-remote.txt).
    """
    import subprocess
    import tempfile

    venv_python = ROOT / "venv-qwen" / "bin" / "python"
    if not venv_python.exists():
        raise RuntimeError(
            "venv-qwen non trovato — esegui prima 'bash setup_qwen.sh' "
            "sull'istanza (vedi QWEN_GGUF_SETUP.md)."
        )
    engine_script = ROOT / "src" / "qwen_engine.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_path = tmp / "originale.png"
        mask_path = tmp / "mask.png"
        out_path = tmp / "risultato.png"
        cv2.imwrite(str(in_path), rgb)
        cv2.imwrite(str(mask_path), md)

        debug_dir = ROOT / "output" / "qwen_debug"
        cmd = [str(venv_python), str(engine_script), str(in_path), str(mask_path),
               str(out_path), "--quant", quant, "--steps", str(steps), "--garment", garment,
               "--debug-dir", str(debug_dir)]
        log(f"  Qwen (subprocess venv-qwen): quant={quant} steps={steps}")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        # encoding esplicito: la barra di progresso di huggingface_hub/tqdm
        # usa caratteri unicode (es. "━"), e la locale di default su alcune
        # istanze Vast e' ASCII puro — senza questo il decode del subprocess
        # crasha (UnicodeDecodeError) invece di far fallire solo Qwen.
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env)
        for line in proc.stdout.splitlines():
            log(f"  [qwen] {line}")
        if proc.returncode != 0:
            for line in proc.stderr.splitlines()[-20:]:
                log(f"  [qwen:err] {line}")
            raise RuntimeError(f"Qwen ha fallito (exit {proc.returncode}) — vedi log sopra")

        res = cv2.imread(str(out_path))
        if res is None:
            raise RuntimeError("Qwen non ha prodotto un file di output leggibile")
        log(f"  Debug Qwen (crop input/mask/output grezzo): {debug_dir}")
        return res


def inpaint_iopaint(rgb, md, engine_key, steps=30):
    """Inpainting tramite IOPaint, che integra e mantiene diversi modelli di
    rimozione con una sola interfaccia. Gestisce da solo il tiling per
    l'alta risoluzione (HDStrategy.CROP), cosa che a mano ci era costata
    parecchio con ZITS."""
    import torch
    from iopaint.model import models
    from iopaint.schema import InpaintRequest, HDStrategy

    name = IOPAINT_ENGINES[engine_key]
    # Un modello alla volta: tenerne piu' d'uno in VRAM insieme a SAM 3
    # satura la 3060. Cambiando motore il precedente viene scaricato.
    for k in list(_IOPAINT):
        if k != name:
            _IOPAINT.pop(k)
    free_vram()
    if name not in _IOPAINT:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # I modelli diffusion (PowerPaint) richiedono un ModelInfo esplicito;
        # quelli "erase" (lama, mat, migan...) no.
        if "/" in name:
            from iopaint.schema import ModelInfo, ModelType
            # L'offload su CPU permette di stare nei 12 GB ma e' DRAMMATICO
            # per la velocita': ~62 s/step contro ~1 s. Su una GPU da 24 GB
            # va disattivato (VRAM_GB sotto), e PowerPaint scende da mezz'ora
            # a un paio di minuti.
            # NIENTE offload su CPU: PowerPaint pesa ~2 GB su disco (fp32),
            # circa 1.2 GB in fp16 sulla GPU, e SAM 3 viene scaricato prima
            # del fill — la VRAM c'e' sia sulla 3060 che sulla 4090.
            # L'offload rimbalzava i pesi su PCIe a ogni passo: 62 s/step
            # contro ~1 s, cioe' mezz'ora per immagine invece di un minuto.
            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / 1e9
                log(f"  VRAM libera: {free_gb:.1f} GB, offload disattivato")
            _IOPAINT[name] = models[name](
                device=dev,
                model_info=ModelInfo(name=name, path=name,
                                     model_type=ModelType.DIFFUSERS_SD_INPAINT),
                enable_controlnet=False,
                low_mem=False,
                disable_nsfw=True,
                cpu_offload=False,
                sd_cpu_textencoder=False,   # richiesto da PowerPaint
                controlnet_method=None,
            )
            # taglia il picco dell'attenzione (che e' cio' che manda in OOM)
            # senza il costo del CPU offload
            pipe = getattr(_IOPAINT[name], "model", None)
            if pipe is not None:
                for fn in ("enable_attention_slicing", "enable_vae_slicing",
                           "enable_vae_tiling"):
                    try:
                        getattr(pipe, fn)()
                    except Exception:
                        pass
        else:
            _IOPAINT[name] = models[name](device=dev)
    model = _IOPAINT[name]

    import torch as _t
    total_gb = (_t.cuda.get_device_properties(0).total_memory / 1e9
                if _t.cuda.is_available() else 0)
    # Limite di risoluzione per i diffusion. Non e' solo VRAM: PowerPaint e'
    # basato su SD 1.5, addestrato a 512px — oltre ~1024 la qualita' PEGGIORA
    # (ripetizioni, artefatti) anche avendo memoria a volonta'. Su 24 GB si
    # puo' alzare, ma non ha senso arrivare a piena risoluzione.
    limit = 1536 if total_gb >= 20 else 1024

    kw = dict(
        hd_strategy=HDStrategy.CROP,          # ritaglia attorno alla maschera
        hd_strategy_crop_trigger_size=1280,
        hd_strategy_crop_margin=196,
        hd_strategy_resize_limit=2048,
    )
    if engine_key == "iop_powerpaint":
        from iopaint.schema import PowerPaintTask
        # I diffusion allocano MOLTO piu' del proprio peso: PowerPaint occupa
        # 2.17 GB ma su un crop 2048px il picco arriva a 9.34 GB (tensori
        # latenti + attenzione). Su 12 GB va in OOM, quindi si lavora a
        # risoluzione piu' bassa e si ricampiona. Su 24 GB si puo' alzare.
        log(f"  diffusion a {limit}px (GPU {total_gb:.0f} GB); il resto del "
            f"capo resta a piena risoluzione")
        # seed fisso (non -1/random): le passate vanno nel registro
        # sessioni.jsonl per capire dove l'automatico sbaglia, un seed
        # casuale renderebbe lo stesso input non riproducibile tra due run.
        sd_seed = 12345
        kw.update(
            prompt="", negative_prompt="",
            powerpaint_task=PowerPaintTask.object_remove,  # cancella, non genera
            sd_steps=steps, sd_guidance_scale=7.5, sd_seed=sd_seed,
        )
        log(f"  PowerPaint: task object-remove, {steps} passi, seed {sd_seed}")
    req = InpaintRequest(**kw)
    work_rgb, work_md = rgb, md
    if engine_key == "iop_powerpaint":
        # PowerPaint elabora alla risoluzione dell'immagine che riceve
        # (height=img_h, width=img_w nel suo forward): le strategie HD di
        # IOPaint non lo riducono. Su 3000x5000 il picco arriva a 9.3 GB e
        # va in OOM. Si ritaglia attorno alla maschera e si riduce a `limit`,
        # poi si reincolla solo l'area mascherata: il resto del capo resta
        # intatto a piena risoluzione.
        ys, xs = np.where(md > 0)
        pad = 200
        y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
        crop_rgb, crop_md = rgb[y0:y1, x0:x1], md[y0:y1, x0:x1]
        ch, cw = crop_rgb.shape[:2]
        sc = min(1.0, limit / max(ch, cw))
        if sc < 1.0:
            nw, nh = (int(cw * sc) // 8) * 8, (int(ch * sc) // 8) * 8
            work_rgb = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            work_md = cv2.resize(crop_md, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            work_rgb, work_md = crop_rgb, crop_md
        log(f"  crop {cw}x{ch} -> elaborato a {work_rgb.shape[1]}x{work_rgb.shape[0]}")

    res = model(work_rgb, work_md, req)        # BGR in, BGR out
    res = np.asarray(res, dtype=np.uint8)

    if engine_key == "iop_powerpaint":
        if res.shape[:2] != (ch, cw):
            res = cv2.resize(res, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        full = rgb.copy()
        sel = crop_md > 0
        region = full[y0:y1, x0:x1]
        region[sel] = res[sel]
        full[y0:y1, x0:x1] = region
        return full
    if res.shape[:2] != rgb.shape[:2]:
        res = cv2.resize(res, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    out = rgb.copy()
    sel = md > 0
    out[sel] = res[sel]                        # fuori maschera resta l'originale
    return out


_LAMA = None


def get_lama():
    """LaMa e' il motore di default: viene invocato a ogni passata, quindi
    va tenuto in cache invece di ricaricarlo da disco ogni volta (prima
    veniva ricreato a ogni chiamata di fill_v3 — lento e affidava al GC lo
    scarico della VRAM, invece che a un unload esplicito)."""
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
    return _LAMA


_ZITS = None


def inpaint_zits(rgb, md):
    """ZITS: ricostruisce prima la STRUTTURA (linee/wireframe via LSM-HAWP +
    transformer), poi la texture. A differenza di LaMa capisce che un bordo o
    una costina continua dietro l'occlusore, invece di limitarsi a estendere
    il colore circostante.

    Vincolo del modello: lavora su immagini QUADRATE e a lato >= 256. Si
    ritaglia un quadrato attorno alla maschera, lo si elabora e lo si
    reinserisce nell'originale a piena risoluzione.
    """
    global _ZITS
    import torch
    free_vram()  # SAM 3 ha appena girato: libera prima di allocare
    zroot = ROOT / "modelli" / "zits_repo"
    if not (zroot / "ckpt" / "best_lsm_hawp.pth").exists():
        raise FileNotFoundError("Pesi ZITS mancanti in modelli/zits_repo/ckpt/")

    if str(zroot) not in sys.path:
        sys.path.insert(0, str(zroot))
    cwd = Path.cwd()
    os.chdir(zroot)  # il codice ZITS usa percorsi relativi ('./ckpt/...')
    try:
        if _ZITS is None:
            from src.config import Config
            from src.FTR_trainer import ZITS as ZITSModel
            from src.lsm_hawp.detector import WireframeDetector
            cfg = Config(str(zroot / "config_scout.yml"))
            cfg.MODE = 1; cfg.gpus = 1; cfg.GPU_ids = "0"; cfg.world_size = 1
            cfg.PATH = "./ckpt/zits_places2_hr"
            model = ZITSModel(cfg, 0, 0, True, True)
            model.inpaint_model.eval()
            wf = WireframeDetector(is_cuda=True).to(0)
            wf.load_state_dict(torch.load("./ckpt/best_lsm_hawp.pth",
                                          map_location="cpu")["model"])
            wf.eval()
            _ZITS = (model, wf)
        model, wf = _ZITS

        import single_image_test as sit
        H, W = md.shape
        ys, xs = np.where(md > 0)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        span = max(ys.max() - ys.min(), xs.max() - xs.min())
        side = min(H, W, span + 900)
        # ZITS alloca ~4 GB su un crop 2912px: oltre ~2000px va in
        # out-of-memory su una 3060 12GB (con SAM 3 gia' in memoria). Si
        # limita il lato e si elabora a scala ridotta, reinserendo il
        # risultato ricampionato solo dentro la maschera.
        MAX_SIDE = 2000
        scale = 1.0
        if side > MAX_SIDE:
            scale = MAX_SIDE / side
        side = max(256, (side // 8) * 8)
        y0 = max(0, min(H - side, cy - side // 2))
        x0 = max(0, min(W - side, cx - side // 2))

        tmp = zroot / "_scout_tmp"
        tmp.mkdir(exist_ok=True)
        crop_img = rgb[y0:y0 + side, x0:x0 + side]
        crop_msk = md[y0:y0 + side, x0:x0 + side]
        work = side
        if scale < 1.0:
            work = max(256, (int(side * scale) // 8) * 8)
            crop_img = cv2.resize(crop_img, (work, work), interpolation=cv2.INTER_AREA)
            crop_msk = cv2.resize(crop_msk, (work, work), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(tmp / "in.png"), crop_img)
        cv2.imwrite(str(tmp / "in_mask.png"), crop_msk)
        sit.test(model, wf, str(tmp / "in.png"), str(tmp / "in_mask.png"),
                 str(tmp), 0.85, sigma256=3.0)

        res = cv2.imread(str(tmp / "in.png"))
        out = rgb.copy()
        if res is not None:
            if res.shape[:2] != (side, side):
                # riporta alla scala piena se si e' lavorato ridotti
                res = cv2.resize(res, (side, side), interpolation=cv2.INTER_LANCZOS4)
            sub = md[y0:y0 + side, x0:x0 + side] > 0
            region = out[y0:y0 + side, x0:x0 + side]
            region[sub] = res[sub]   # solo dentro la maschera: il resto intatto
            out[y0:y0 + side, x0:x0 + side] = region
        return out, side
    finally:
        os.chdir(cwd)


def fill_v3(rgb, mask, dilate_px, alpha=None, engine="lama", sd_steps=30, patch_min_tessuto=0.97):
    """Logica v3: dilatazione uniforme leggera + LaMa sul crop.

    v3 e' la versione che nei confronti misurati danneggiava meno il capo.
    Le versioni successive (split per componente + dilatazione BG 45px)
    riducevano il residuo ma mangiavano il bordo del capo: regressione.
    """
    k = np.ones((dilate_px, dilate_px), np.uint8)
    md = cv2.dilate(mask, k, iterations=1)

    rgb_out = rgb.copy()
    # Se abbiamo l'alpha della macchina: dove l'alpha dice "sfondo", riempi
    # direttamente di bianco senza scomodare LaMa. Piu' sicuro e istantaneo.
    if alpha is not None:
        bg = (alpha < 128) & (md > 0)
        rgb_out[bg] = 255
        md = md.copy()
        md[bg] = 0

    if (md > 0).sum() == 0:
        return rgb_out, md, 0

    # "none": l'area della gruccia resta bianca, buco netto e visibile.
    # Serve sui tessuti a fantasia, dove l'inpainting deve inventare un
    # pattern figurativo e sbaglia in modo evidente: meglio un buco pulito
    # da chiudere in Photoshop clonando il pattern vero dello stesso capo.
    if engine == "none":
        rgb_out[md > 0] = 255
        return rgb_out, md, 0

    # "patch": clona il pattern vero dello stesso capo, poi LaMa rifinisce
    # solo i bordi residui. Pensato per i tessuti a fantasia, dove estendere
    # la texture (LaMa) o inventarla (generativi) falliscono entrambi.
    if engine == "patch":
        out, applied = patch_fill(rgb_out, md, min_tessuto=patch_min_tessuto)
        return out, md, applied

    if engine == "zits":
        out, side = inpaint_zits(rgb_out, md)
        return out, md, side

    if engine in IOPAINT_ENGINES:
        return inpaint_iopaint(rgb_out, md, engine, sd_steps), md, 0

    if engine == "qwen":
        return inpaint_qwen(rgb_out, md, steps=sd_steps), md, 0

    ys, xs = np.where(md > 0)
    margin = 250
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin)
    crop = cv2.cvtColor(rgb_out[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    cmask = md[y0:y1, x0:x1]

    from PIL import Image as PILImage
    lama = get_lama()
    res = np.array(lama(PILImage.fromarray(crop), PILImage.fromarray(cmask)))
    if res.shape[:2] != crop.shape[:2]:
        res = cv2.resize(res, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    region = rgb_out[y0:y1, x0:x1]
    sel = cmask > 0
    region[sel] = res_bgr[sel]        # fuori maschera: pixel del crop originale intatti
    rgb_out[y0:y1, x0:x1] = region
    return rgb_out, md, 0


# -------------------------------------------------------------------- job

# Motori con possibilita' reale di restare in uso (vedi consulto in pareri/).
# Gli altri (SD1.5, IOPaint MAT/MIGAN/FcF/ZITS/LDM/LaMa, GroundingDINO+SAM2)
# sono stati tolti: mai indicati come promettenti, solo termini di paragone.
# "qwen": test esplorativo (Qwen Image Inpaint GGUF), gira in venv-qwen via
# subprocess — vedi inpaint_qwen() e QWEN_GGUF_SETUP.md.
ENGINES = Literal["lama", "iop_powerpaint", "zits", "patch", "none", "qwen"]


class RunReq(BaseModel):
    filename: str
    detector: Literal["sam3"] = "sam3"
    prompts: str = "clothes hanger, hook, clip"
    conf: float = Field(default=0.25, ge=0.05, le=0.9)
    dilate_px: int = Field(default=15, ge=1, le=101)  # default v3
    use_alpha: bool = True
    clean_frags: bool = True        # togli nastro adesivo / frammenti isolati
    # Lavoro iterativo: se continue_session, si parte dal risultato della
    # passata precedente invece che dall'originale. `zona` limita la ricerca
    # a un rettangolo [x0,y0,x1,y1] in coordinate relative 0-1.
    continue_session: bool = False
    zona: list[float] | None = None
    # Se True e c'e' una zona: la zona STESSA diventa la maschera, SAM 3 non
    # viene interpellato. Serve per rifinire un residuo di TEXTURE (fill
    # sbagliato della passata precedente) invece di un oggetto fisico —
    # li' SAM 3 non trova nulla da segmentare col prompt (giustamente: non
    # c'e' piu' ne' gancio ne' clip), quindi la passata falliva sempre.
    zona_e_maschera: bool = False
    nota: str = ""                  # annotazione libera dell'operatore
    catch_shadows: bool = True      # estendi la maschera alle ombre della gruccia
    shadow_strength: float = Field(default=0.82, ge=0.5, le=0.98)
    engine: ENGINES = "lama"
    sd_steps: int = Field(default=30, ge=4, le=60)  # passi diffusione (PowerPaint)
    # soglia "quanto tessuto pulito" per patch_fill: vicino a un bordo curvo
    # o stretto va abbassata, altrimenti la ricerca fallisce sempre e la
    # componente resta senza fill (vedi patch_fill).
    patch_min_tessuto: float = Field(default=0.97, ge=0.5, le=0.99)


def run_job(req: RunReq):
    try:
        raw = UPLOADS.get(req.filename)
        if raw is None:
            raise FileNotFoundError(
                f"'{req.filename}' non è più in memoria — ricaricala trascinandola."
            )
        rgb, alpha = load_rgb(raw)
        # Passata successiva sullo stesso capo: si riparte dal risultato
        # accumulato, non dall'originale. Il residuo e' piu' piccolo e
        # circondato da piu' contesto pulito, quindi il fill ha vita facile.
        n_pass = 1
        if (req.continue_session and SESSION["current"] is not None
                and SESSION["filename"] == req.filename):
            rgb = SESSION["current"].copy()
            n_pass = len(SESSION["passes"]) + 1
            log(f"Passata {n_pass} su {req.filename} (riparte dal risultato precedente)")
        else:
            SESSION.update({"filename": req.filename, "current": None, "passes": []})
            log(f"Carico {req.filename} — passata 1")
        log(f"  risoluzione {rgb.shape[1]}x{rgb.shape[0]}, alpha: {'si' if alpha is not None else 'no'}")

        t0 = time.time()
        H, W = rgb.shape[:2]
        prompts = None   # nessuna detection testuale quando zona_e_maschera

        if req.zona_e_maschera:
            # Bypassa SAM 3: il rettangolo disegnato dall'operatore E' la
            # maschera. Per rifinire un residuo di texture, non un oggetto.
            if not req.zona or len(req.zona) != 4:
                raise RuntimeError(
                    "Serve una zona disegnata sul risultato per usarla come maschera.")
            zx0, zy0, zx1, zy1 = (max(0.0, min(1.0, v)) for v in req.zona)
            zx0, zx1 = sorted((zx0, zx1))
            zy0, zy1 = sorted((zy0, zy1))
            mask = np.zeros((H, W), np.uint8)
            mask[int(zy0 * H):int(zy1 * H), int(zx0 * W):int(zx1 * W)] = 255
            log(f"  zona usata come maschera diretta: {int((mask > 0).sum())} px, SAM 3 saltato")
        else:
            prompts = [p.strip() for p in req.prompts.split(",") if p.strip()]
            if not prompts:
                raise RuntimeError("Prompt vuoto: indica almeno una parola chiave (es. 'clothes hanger').")
            log(f"Detection SAM 3, prompt={prompts} conf={req.conf}")
            mask, info = detect_sam3(rgb, prompts, req.conf)
            if mask is None:
                raise RuntimeError("Nessuna maschera trovata — prova ad abbassare conf o cambiare prompt.")
            log(f"  maschera: {int((mask>0).sum())} px  ({time.time()-t0:.1f}s)  {info}")

            # Zona ristretta: l'operatore ha selezionato un rettangolo attorno
            # al residuo, quindi si ignora cio' che SAM 3 trova fuori da li'.
            if req.zona:
                if len(req.zona) != 4:
                    raise RuntimeError("Zona non valida: servono 4 coordinate [x0,y0,x1,y1].")
                zx0, zy0, zx1, zy1 = (max(0.0, min(1.0, v)) for v in req.zona)
                zx0, zx1 = sorted((zx0, zx1))
                zy0, zy1 = sorted((zy0, zy1))
                box = np.zeros_like(mask)
                box[int(zy0 * H):int(zy1 * H), int(zx0 * W):int(zx1 * W)] = 255
                before = int((mask > 0).sum())
                mask = cv2.bitwise_and(mask, box)
                log(f"  zona selezionata: maschera ristretta da {before} a "
                    f"{int((mask > 0).sum())} px")
                if (mask > 0).sum() == 0:
                    raise RuntimeError(
                        "Nessuna gruccia trovata dentro la zona selezionata — "
                        "prova ad allargarla o ad abbassare la confidenza.")

        if req.catch_shadows and not req.zona_e_maschera:
            mask, nsh = add_shadows(rgb, mask, strength=req.shadow_strength)
            log(f"Ombre della gruccia sul tessuto: +{nsh} px aggiunti alla maschera")

        names = {"lama": "LaMa", "patch": "Texture patching (clona il pattern del capo)",
                 "zits": "ZITS (struttura + texture)", "none": "nessun fill (buco bianco)",
                 "iop_powerpaint": "PowerPaint (task object-remove)",
                 "qwen": "Qwen Image Inpaint (GGUF, esplorativo)"}
        dilate_px = req.dilate_px | 1   # kernel dispari: centratura simmetrica
        log(f"Fill: {names.get(req.engine, req.engine)} — dilate {dilate_px}px, "
            f"alpha={'si' if req.use_alpha and alpha is not None else 'no'}")
        t1 = time.time()
        out, md, npatch = fill_v3(rgb, mask, dilate_px, alpha if req.use_alpha else None,
                          engine=req.engine, sd_steps=req.sd_steps,
                          patch_min_tessuto=req.patch_min_tessuto)
        log(f"  completato in {time.time()-t1:.1f}s")
        if req.engine == "patch":
            log(f"  patch di tessuto clonate dal capo: {npatch} (soglia tessuto {req.patch_min_tessuto})")
        elif req.engine == "zits":
            log(f"  elaborato su crop quadrato {npatch}x{npatch} px")

        frag_mask = np.zeros(rgb.shape[:2], bool)
        if req.clean_frags:
            before_frag = out.copy()
            out, nfrag, sizes = clean_isolated(out, alpha)
            # i px tolti qui sono rimozioni VOLUTE (nastro), non danno al capo:
            # vanno scontati dalla metrica, altrimenti la falsano
            frag_mask = (np.abs(out.astype(int) - before_frag.astype(int)).max(axis=2) > 0)
            if nfrag:
                log(f"Frammenti isolati rimossi (nastro/spilli): {nfrag} — aree {sizes}")
            else:
                log("Nessun frammento isolato da rimuovere")

        # Niente scrittura su disco: il PNG pieno resta in memoria server-side
        # e viene servito da /api/download solo su richiesta esplicita —
        # prima finiva dentro /api/status, scaricato per intero a ogni poll.
        stem = Path(req.filename).stem
        ok, buf = cv2.imencode(".png", out)
        full_png_bytes = buf.tobytes() if ok else None
        DOWNLOAD["png"] = full_png_bytes
        DOWNLOAD["name"] = f"{stem}_pulita.png"

        ov = rgb.copy()
        ov[md > 0] = (0, 0, 255)
        ov = cv2.addWeighted(rgb, 0.55, ov, 0.45, 0)

        # px del capo alterati: la metrica che ha rivelato la regressione v15.
        # Conta i pixel cambiati FUORI dalla maschera dilatata — cioe' il
        # danno collaterale sul capo, che e' cio' che si nota a occhio.
        # Esclude anche l'area fuori-soggetto (dove l'alpha dice sfondo): li'
        # ogni modifica e' voluta (nastro, frammenti), non danno al capo.
        diff = cv2.absdiff(out, rgb).max(axis=2) > 18
        outside = (alpha < 128) if alpha is not None else np.zeros(diff.shape, bool)
        collateral = int((diff & (md == 0) & ~outside & ~frag_mask).sum())

        with _lock:
            JOB["result"] = {
                "before": to_data_uri(rgb),
                "after": to_data_uri(out),
                "overlay": to_data_uri(ov),
                "mask_px": int((md > 0).sum()),
                "collateral_px": collateral,
                "has_download": full_png_bytes is not None,
                "download_name": DOWNLOAD["name"],
                "pass_n": n_pass,
            }
        log(f"px capo alterati fuori maschera: {collateral}  (piu' basso = meglio)")

        # Il risultato diventa la base della prossima passata, e la passata
        # viene registrata: e' il dato che dice dove l'automatico sbaglia.
        SESSION["current"] = out
        rec = {
            "capo": req.filename, "passata": n_pass,
            "detector": req.detector, "engine": req.engine,
            "prompt": prompts, "conf": req.conf, "dilate_px": dilate_px,
            "zona": req.zona, "nota": req.nota,
            "use_alpha": req.use_alpha,
            "catch_shadows": req.catch_shadows,
            "shadow_strength": req.shadow_strength if req.catch_shadows else None,
            "mask_px": int((md > 0).sum()), "collateral_px": collateral,
            "secondi": round(time.time() - t0, 1),
        }
        SESSION["passes"].append(rec)
        save_pass(rec)
        log(f"Passata {n_pass} registrata in output/sessioni.jsonl")
    except Exception as e:
        import traceback
        log(f"ERRORE: {type(e).__name__}: {e}")
        with _lock:
            JOB["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        with _lock:
            JOB["running"] = False


@app.get("/api/images")
async def api_images():
    ck = ROOT / "modelli" / "sam3" / "sam3.pt"
    return {
        "images": sorted(UPLOADS.keys()),
        "sam3_ready": ck.exists() and ck.stat().st_size > 1_000_000,
        "sam3_size_mb": round(ck.stat().st_size / 1e6, 1) if ck.exists() else 0,
    }


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    """Riceve le immagini trascinate e le tiene in memoria per la sessione.
    Non scrive nulla su disco: al refresh la lista riparte vuota."""
    saved = []
    for f in files:
        name = Path(f.filename).name
        if Path(name).suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
            continue
        UPLOADS[name] = await f.read()
        saved.append(name)
    return {"saved": saved}


def save_pass(entry: dict):
    """Registra una passata su file, in append. Una riga JSON per passata:
    serve a capire dove la parte automatica sbaglia sistematicamente
    (quante passate servono, in che zona, con quali parametri)."""
    import datetime
    entry["quando"] = datetime.datetime.now().isoformat(timespec="seconds")
    SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SESSION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.get("/api/session")
async def api_session():
    """Stato della sessione iterativa in corso."""
    return {
        "filename": SESSION["filename"],
        "passes": SESSION["passes"],
        "has_current": SESSION["current"] is not None,
    }


@app.post("/api/session/reset")
async def api_session_reset():
    """Riparte dall'immagine originale, scartando le passate fatte."""
    SESSION.update({"filename": None, "current": None, "passes": []})
    return {"ok": True}


# ----------------------------------------------------------------- warping

class Tratto(BaseModel):
    """Un trascinamento: prendi il tessuto in (x0,y0), portalo in (x1,y1).
    Coordinate RELATIVE 0-1, così la pagina può lavorare sull'anteprima
    ridotta e il server applicare a piena risoluzione."""
    x0: float
    y0: float
    x1: float
    y1: float
    raggio: float = 0.12            # frazione del lato lungo dell'immagine
    forza: float = 1.0


class Zona(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class WarpReq(BaseModel):
    filename: str
    # modo "tratti": trascinamenti liberi (il Liquify vero e proprio)
    tratti: list[Tratto] = []
    # modo "linea": due punti da portare in orizzontale (o verticale)
    linea: list[float] | None = None      # [x0,y0,x1,y1] relative
    # modo "curva": N punti (>=2) portati tutti alla stessa quota. È il caso
    # reale più frequente — la vita tirata dalle clip forma una V, e con due
    # soli punti si ruoterebbe la V invece di appiattirla.
    punti: list[list[float]] = []         # [[x,y], ...] relative
    orizzontale: bool = True
    fascia: float = 0.0                   # 0 = automatica dalla lunghezza
    congelate: list[Zona] = []
    nota: str = ""


def _immagine_di_lavoro(filename: str):
    """L'immagine su cui lavorare: il risultato della rimozione gruccia se
    c'e', altrimenti l'originale.

    Il warping va misurato DOPO la rimozione: sulla foto grezza la sagoma
    include gancio e staffa, e si finisce per misurare la ferramenta invece
    del capo (verificato: angoli di +60° su capi che pendono di 2°).
    """
    if SESSION["current"] is not None and SESSION["filename"] == filename:
        return SESSION["current"].copy(), None, True
    raw = UPLOADS.get(filename)
    if raw is None:
        raise FileNotFoundError(
            f"'{filename}' non è più in memoria — ricaricala trascinandola.")
    rgb, alpha = load_rgb(raw)
    return rgb, alpha, False


@app.get("/api/warp/immagine")
async def api_warp_immagine(filename: str):
    """L'immagine su cui l'operatore disegnerà i tratti.

    È il risultato della rimozione gruccia se c'è: il warp si fa dopo, su un
    capo già pulito.
    """
    try:
        rgb, alpha, da_sessione = _immagine_di_lavoro(filename)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    h, w = rgb.shape[:2]
    return {
        "img": to_data_uri(rgb),
        "w": w, "h": h,
        "da_sessione": da_sessione,
    }


@app.post("/api/warp/applica")
async def api_warp_applica(req: WarpReq):
    """Applica la deformazione indicata a mano e restituisce prima/dopo.

    Le coordinate arrivano relative (0-1) e vengono scalate qui alla
    risoluzione piena: così la pagina lavora su un'anteprima leggera ma la
    correzione è applicata sull'originale senza perdita.
    """
    from scout_warp import applica_tratti, applica_linea, applica_curva

    try:
        rgb, alpha, _ = _immagine_di_lavoro(req.filename)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    h, w = rgb.shape[:2]
    lato = float(max(h, w))
    congelate = [{"x0": z.x0 * w, "y0": z.y0 * h, "x1": z.x1 * w, "y1": z.y1 * h}
                 for z in req.congelate]

    t0 = time.time()
    try:
        if req.punti and len(req.punti) >= 2:
            pts = [(p[0] * w, p[1] * h) for p in req.punti]
            fascia = req.fascia * lato if req.fascia else None
            out, out_a, info = applica_curva(
                rgb, pts, alpha, altezza_fascia=fascia, congelate=congelate)
            operazione = "warp_curva"
        elif req.linea and len(req.linea) == 4:
            x0, y0, x1, y1 = req.linea
            fascia = req.fascia * lato if req.fascia else None
            out, out_a, info = applica_linea(
                rgb, (x0 * w, y0 * h), (x1 * w, y1 * h), alpha,
                altezza_fascia=fascia, congelate=congelate,
                orizzontale=req.orizzontale)
            operazione = "warp_linea"
        elif req.tratti:
            tratti = [{"x0": t.x0 * w, "y0": t.y0 * h,
                       "x1": t.x1 * w, "y1": t.y1 * h,
                       "raggio": t.raggio * lato, "forza": t.forza}
                      for t in req.tratti]
            out, out_a, info = applica_tratti(rgb, tratti, alpha, congelate)
            operazione = "warp_tratti"
        else:
            return JSONResponse(
                {"error": "Nessuna correzione indicata: clicca almeno due "
                          "punti sulla linea, o traccia un trascinamento."},
                status_code=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    SESSION["filename"] = req.filename
    SESSION["current"] = out

    rec = {
        "capo": req.filename, "operazione": operazione,
        "tratti": info.get("tratti", 0),
        "spostamento_max_px": info.get("spostamento_max", 0.0),
        "congelate": len(congelate),
        "nota": req.nota, "secondi": round(time.time() - t0, 1),
    }
    SESSION["passes"].append(rec)
    save_pass(rec)

    ok, buf = cv2.imencode(".png", out)
    stem = Path(req.filename).stem
    return {
        "before": to_data_uri(rgb),
        "after": to_data_uri(out),
        "info": info,
        "secondi": rec["secondi"],
        "download": ("data:image/png;base64," + base64.b64encode(buf).decode()) if ok else None,
        "download_name": f"{stem}_corretta.png",
    }


@app.post("/api/reset")
async def api_reset():
    """Svuota le immagini in memoria: la pagina lo chiama a ogni caricamento,
    così un refresh riparte sempre da zero — inclusa la sessione iterativa,
    altrimenti una nuova immagine con lo stesso nome di una vecchia poteva
    ripartire dal risultato della sessione precedente."""
    with _lock:
        if JOB["running"]:
            # un refresh durante l'elaborazione non deve buttare via
            # l'immagine che il job sta usando
            return {"cleared": 0, "skipped": "elaborazione in corso"}
    n = len(UPLOADS)
    UPLOADS.clear()
    SESSION.update({"filename": None, "current": None, "passes": []})
    DOWNLOAD.update({"png": None, "name": None})
    with _lock:
        JOB.update({"running": False, "log": [], "result": None, "error": None})
    return {"cleared": n}


@app.post("/api/run")
async def api_run(req: RunReq):
    with _lock:
        if JOB["running"]:
            return JSONResponse({"error": "Elaborazione già in corso"}, status_code=409)
        JOB.update({"running": True, "log": [], "result": None, "error": None})
    threading.Thread(target=run_job, args=(req,), daemon=True).start()
    return {"started": True}


@app.get("/api/status")
async def api_status():
    with _lock:
        return dict(JOB)


CONFRONTI_DIR = ROOT / "output" / "confronti"


@app.post("/api/salva_confronto")
async def api_salva_confronto():
    """Salva su disco l'ultimo risultato con un banner che riporta i
    parametri usati (dilatazione, alpha, ombre...) stampato sull'immagine:
    pensato per confrontare piu' salvataggi fatti cambiando un solo
    parametro alla volta (es. use_alpha ON/OFF) senza doverli etichettare
    a mano o tenere a mente quale era quale."""
    if DOWNLOAD["png"] is None:
        return JSONResponse({"error": "Nessun risultato disponibile"}, status_code=404)
    if not SESSION["passes"]:
        return JSONResponse({"error": "Nessuna passata registrata per questo risultato"}, status_code=404)

    rec = SESSION["passes"][-1]
    img = cv2.imdecode(np.frombuffer(DOWNLOAD["png"], np.uint8), cv2.IMREAD_COLOR)
    etichettata = banner_parametri(img, rec)

    CONFRONTI_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(rec.get("capo", "capo")).stem
    quando = rec.get("quando", "").replace(":", "-")
    nome = f"{stem}_p{rec.get('passata', 1)}_{quando}.png"
    out_path = CONFRONTI_DIR / nome
    cv2.imwrite(str(out_path), etichettata)

    return {"salvato": str(out_path.relative_to(ROOT))}


@app.get("/api/download")
async def api_download():
    """PNG a piena risoluzione dell'ultimo risultato, servito solo su
    richiesta esplicita (bottone Scarica) — non piu' incluso nel polling
    di /api/status, che restava pesante anche a job concluso."""
    if DOWNLOAD["png"] is None:
        return JSONResponse({"error": "Nessun risultato disponibile"}, status_code=404)
    from fastapi.responses import Response
    return Response(
        content=DOWNLOAD["png"], media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{DOWNLOAD["name"]}"'},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    print("GUI controllo gruccia:  http://127.0.0.1:8095")
    uvicorn.run(app, host="127.0.0.1", port=8095, log_level="warning")
