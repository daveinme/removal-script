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
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).parent.parent
SCOUT = ROOT.parent
sys.path.insert(0, str(ROOT))

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


def detect_dino_sam2(rgb, prompt_str, box_thr=0.25, text_thr=0.20):
    """Pipeline storica: GroundingDINO produce le box, SAM2 le segmenta.
    Serve come termine di paragone contro SAM 3."""
    import torch
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    from PIL import Image as PILImage

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ROOT / "venv/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    wts = ROOT / "modelli/groundingdino/groundingdino_swint_ogc.pth"
    model = load_model(str(cfg), str(wts)).to(dev)

    src = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    tf = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    timg, _ = tf(PILImage.fromarray(src), None)
    boxes, logits, phrases = predict(model=model, image=timg, caption=prompt_str,
                                     box_threshold=box_thr, text_threshold=text_thr, device=dev)
    # scarta box enormi: falsi positivi che coprono mezza immagine
    keep = [i for i in range(len(boxes)) if float(boxes[i, 2] * boxes[i, 3]) <= 0.25]
    boxes = boxes[keep]
    if len(boxes) == 0:
        return None, []
    h, w = rgb.shape[:2]
    xyxy = boxes.clone()
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
                     str(ROOT / "modelli/sam2/sam2.1_hiera_large.pt"), device=dev)
    p = SAM2ImagePredictor(sam)
    p.set_image(src)
    masks, _, _ = p.predict(box=xyxy.numpy(), multimask_output=False)
    comb = np.zeros((h, w), np.uint8)
    for m in masks:
        comb |= (( m.squeeze() if m.ndim > 2 else m) > 0.5).astype(np.uint8) * 255
    return comb, [str(x) for x in phrases]


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


def patch_fill(rgb, md, search_px=None, patch_margin=40):
    """Texture patching: riempie il buco copiando il pattern VERO dallo stesso
    capo, invece di farlo inventare a un modello.

    Per ogni componente della maschera cerca, nell'intorno, una zona di
    tessuto valida (fuori maschera) delle stesse dimensioni e la incolla con
    cv2.seamlessClone (Poisson blending), che raccorda luminosita' e colore
    ai bordi. Il pattern risultante e' quello reale del capo — nessuna
    invenzione, che e' il limite di tutti i generativi provati.

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
                if (win.min(axis=2) < 235).mean() < 0.97:
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

# Motori di rimozione forniti da IOPaint. Sono tutti "erase" (addestrati a
# cancellare, non a generare) e girano sulla 3060.
IOPAINT_ENGINES = {
    "iop_lama":  "lama",    # stessa rete del nostro LaMa, ma con tiling IOPaint
    "iop_mat":   "mat",     # Mask-Aware Transformer, per buchi grandi
    "iop_migan": "migan",   # leggero e veloce
    "iop_fcf":   "fcf",     # Fourier CNN, buono sulle strutture
    "iop_zits":  "zits",    # struttura + texture (versione mantenuta)
    "iop_ldm":   "ldm",     # latent diffusion, piu' lento
    # PowerPaint e' l'unico addestrato SPECIFICAMENTE per l'object removal:
    # ha un task dedicato ("object-remove") invece di un prompt generico.
    # E' un diffusion, ma guidato a cancellare e non a generare — quindi in
    # teoria non dovrebbe ricadere nel difetto di SD 1.5 (che inventava
    # una gruccia al posto di quella da togliere).
    "iop_powerpaint": "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
}


def inpaint_iopaint(rgb, md, engine_key):
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
            _IOPAINT[name] = models[name](
                device=dev,
                model_info=ModelInfo(name=name, path=name,
                                     model_type=ModelType.DIFFUSERS_SD_INPAINT),
                enable_controlnet=False,
                low_mem=True,          # offload: serve sui 12 GB
                disable_nsfw=True,
                cpu_offload=True,
            )
        else:
            _IOPAINT[name] = models[name](device=dev)
    model = _IOPAINT[name]

    kw = dict(
        hd_strategy=HDStrategy.CROP,          # ritaglia attorno alla maschera
        hd_strategy_crop_trigger_size=1280,
        hd_strategy_crop_margin=196,
        hd_strategy_resize_limit=2048,
    )
    if engine_key == "iop_powerpaint":
        from iopaint.schema import PowerPaintTask
        kw.update(
            prompt="", negative_prompt="",
            powerpaint_task=PowerPaintTask.object_remove,  # cancella, non genera
            sd_steps=30, sd_guidance_scale=7.5, sd_seed=-1,
        )
    req = InpaintRequest(**kw)
    res = model(rgb, md, req)                  # BGR in, BGR out
    res = np.asarray(res, dtype=np.uint8)
    if res.shape[:2] != rgb.shape[:2]:
        res = cv2.resize(res, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    out = rgb.copy()
    sel = md > 0
    out[sel] = res[sel]                        # fuori maschera resta l'originale
    return out


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


_SD_PIPE = None


def inpaint_sd15(crop_rgb, crop_mask, prompt=""):
    """Inpainting generativo con Stable Diffusion 1.5 inpainting.

    ATTENZIONE: a differenza di LaMa, questo modello INVENTA il contenuto.
    Su un tessuto a fantasia produce un pattern plausibile ma DIVERSO da
    quello reale del capo. Va usato sapendolo, e verificato caso per caso.

    Lavora a 512x512 (la risoluzione nativa del modello) e poi ricampiona
    alla dimensione del crop: quindi il dettaglio fine viene comunque perso
    se il crop e' molto piu' grande.
    """
    global _SD_PIPE
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image as PILImage

    if _SD_PIPE is None:
        _SD_PIPE = StableDiffusionInpaintPipeline.from_pretrained(
            "botp/stable-diffusion-v1-5-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        _SD_PIPE.set_progress_bar_config(disable=True)
        if torch.cuda.is_available():
            # tiene i pesi in RAM e sposta sulla GPU solo il modulo in uso:
            # convive con SAM 3 senza saturare la VRAM
            _SD_PIPE.enable_model_cpu_offload()
            _SD_PIPE.enable_attention_slicing()
        else:
            _SD_PIPE = _SD_PIPE.to("cpu")

    h, w = crop_rgb.shape[:2]
    img = PILImage.fromarray(crop_rgb).resize((512, 512), PILImage.LANCZOS)
    msk = PILImage.fromarray(crop_mask).resize((512, 512), PILImage.NEAREST)
    if not prompt:
        prompt = "seamless continuation of the surrounding fabric texture, same pattern, product photo"
    out = _SD_PIPE(
        prompt=prompt,
        negative_prompt="hanger, hook, clip, distortion, blurry, artifacts, text, watermark",
        image=img, mask_image=msk,
        num_inference_steps=30, guidance_scale=7.5,
    ).images[0]
    return np.array(out.resize((w, h), PILImage.LANCZOS))


def fill_v3(rgb, mask, dilate_px, alpha=None, use_lama=True, engine="lama", sd_prompt=""):
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
        out, applied = patch_fill(rgb_out, md)
        return out, md, applied

    if engine == "zits":
        out, side = inpaint_zits(rgb_out, md)
        return out, md, side

    if engine in IOPAINT_ENGINES:
        return inpaint_iopaint(rgb_out, md, engine), md, 0

    ys, xs = np.where(md > 0)
    margin = 250
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin)
    crop = cv2.cvtColor(rgb_out[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    cmask = md[y0:y1, x0:x1]

    from PIL import Image as PILImage
    if engine == "sd15":
        res = inpaint_sd15(crop, cmask, sd_prompt)
    else:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        res = np.array(lama(PILImage.fromarray(crop), PILImage.fromarray(cmask)))
    if res.shape[:2] != crop.shape[:2]:
        res = cv2.resize(res, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    merged = np.where((cmask > 0)[:, :, None], res, crop)
    rgb_out[y0:y1, x0:x1] = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
    return rgb_out, md, 0


# -------------------------------------------------------------------- job

class RunReq(BaseModel):
    filename: str
    detector: str = "sam3"          # "sam3" | "dino_sam2"
    prompts: str = "clothes hanger, hook, clip"
    conf: float = 0.25
    dilate_px: int = 15             # default v3
    use_alpha: bool = True
    clean_frags: bool = True        # togli nastro adesivo / frammenti isolati
    catch_shadows: bool = True      # estendi la maschera alle ombre della gruccia
    shadow_strength: float = 0.82   # piu' basso = solo ombre molto marcate
    engine: str = "lama"            # "lama" | "sd15" (generativo) | "none" (buco bianco)
    sd_prompt: str = ""             # prompt libero per sd15; vuoto = descrizione generica


def run_job(req: RunReq):
    try:
        raw = UPLOADS.get(req.filename)
        if raw is None:
            raise FileNotFoundError(
                f"'{req.filename}' non è più in memoria — ricaricala trascinandola."
            )
        log(f"Carico {req.filename}")
        rgb, alpha = load_rgb(raw)
        log(f"  risoluzione {rgb.shape[1]}x{rgb.shape[0]}, alpha: {'si' if alpha is not None else 'no'}")

        prompts = [p.strip() for p in req.prompts.split(",") if p.strip()]
        t0 = time.time()
        if req.detector == "sam3":
            log(f"Detection SAM 3, prompt={prompts} conf={req.conf}")
            mask, info = detect_sam3(rgb, prompts, req.conf)
        else:
            log(f"Detection GroundingDINO + SAM2 (pipeline storica)")
            mask, info = detect_dino_sam2(rgb, " . ".join(prompts))
        if mask is None:
            raise RuntimeError("Nessuna maschera trovata — prova ad abbassare conf o cambiare prompt.")
        log(f"  maschera: {int((mask>0).sum())} px  ({time.time()-t0:.1f}s)  {info}")

        if req.catch_shadows:
            mask, nsh = add_shadows(rgb, mask, strength=req.shadow_strength)
            log(f"Ombre della gruccia sul tessuto: +{nsh} px aggiunti alla maschera")

        names = {"lama": "LaMa", "sd15": "SD 1.5 generativo",
                 "patch": "Texture patching (clona il pattern del capo)",
                 "zits": "ZITS (struttura + texture)",
                 "none": "nessun fill (buco bianco)",
                 "iop_lama": "LaMa via IOPaint", "iop_mat": "MAT via IOPaint",
                 "iop_migan": "MIGAN via IOPaint", "iop_fcf": "FcF via IOPaint",
                 "iop_zits": "ZITS via IOPaint", "iop_ldm": "LDM via IOPaint",
                 "iop_powerpaint": "PowerPaint (task object-remove)"}
        log(f"Fill: {names.get(req.engine, req.engine)} — dilate {req.dilate_px}px, "
            f"alpha={'si' if req.use_alpha and alpha is not None else 'no'}")
        if req.engine == "sd15":
            log("  NB: il generativo INVENTA il contenuto — su una fantasia il")
            log("      pattern sara' plausibile ma diverso da quello reale.")
        t1 = time.time()
        out, md, npatch = fill_v3(rgb, mask, req.dilate_px, alpha if req.use_alpha else None,
                          engine=req.engine, sd_prompt=req.sd_prompt)
        log(f"  completato in {time.time()-t1:.1f}s")
        if req.engine == "patch":
            log(f"  patch di tessuto clonate dal capo: {npatch}")
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

        # Niente scrittura su disco: il risultato torna alla pagina, che offre
        # il download a piena risoluzione se serve conservarlo.
        stem = Path(req.filename).stem
        ok, buf = cv2.imencode(".png", out)
        full_png = "data:image/png;base64," + base64.b64encode(buf).decode() if ok else None

        ov = rgb.copy()
        ov[md > 0] = (0, 0, 255)
        ov = cv2.addWeighted(rgb, 0.55, ov, 0.45, 0)

        # px del capo alterati: la metrica che ha rivelato la regressione v15.
        # Conta i pixel cambiati FUORI dalla maschera dilatata — cioe' il
        # danno collaterale sul capo, che e' cio' che si nota a occhio.
        # Esclude anche l'area fuori-soggetto (dove l'alpha dice sfondo): li'
        # ogni modifica e' voluta (nastro, frammenti), non danno al capo.
        diff = (np.abs(out.astype(int) - rgb.astype(int)).max(axis=2) > 18)
        outside = (alpha < 128) if alpha is not None else np.zeros(diff.shape, bool)
        collateral = int((diff & (md == 0) & ~outside & ~frag_mask).sum())

        with _lock:
            JOB["result"] = {
                "before": to_data_uri(rgb),
                "after": to_data_uri(out),
                "overlay": to_data_uri(ov),
                "mask_px": int((md > 0).sum()),
                "collateral_px": collateral,
                "download": full_png,
                "download_name": f"{stem}_pulita.png",
            }
        log(f"px capo alterati fuori maschera: {collateral}  (piu' basso = meglio)")
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


@app.post("/api/reset")
async def api_reset():
    """Svuota le immagini in memoria: la pagina lo chiama a ogni caricamento,
    così un refresh riparte sempre da zero."""
    with _lock:
        if JOB["running"]:
            # un refresh durante l'elaborazione non deve buttare via
            # l'immagine che il job sta usando
            return {"cleared": 0, "skipped": "elaborazione in corso"}
    n = len(UPLOADS)
    UPLOADS.clear()
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    print("GUI controllo gruccia:  http://127.0.0.1:8095")
    uvicorn.run(app, host="127.0.0.1", port=8095, log_level="warning")
