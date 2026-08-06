#!/usr/bin/env python3
"""
Sito di review locale per il progetto rimozione-gruccia. Legge le immagini in
output/iterazioni/, le sincronizza su R2 (idempotente) e serve una pagina HTML
che le mostra raggruppate per iterazione, con lightbox per l'ingrandimento.
Avvio: python3 server.py (poi nginx/tunnel espongono su gruccia.sitora.it)
"""
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

from sync_r2 import sync  # noqa: E402

PUBLIC_URL = os.environ["R2_PUBLIC_URL"].rstrip("/")
ITER_DIR = ROOT / "output" / "iterazioni"

app = FastAPI(title="Review rimozione gruccia")


def _human_label(tag: str) -> str:
    return re.sub(r"^v(\d+)_", r"v\1 — ", tag).replace("_", " ")


def build_page() -> str:
    # NB: niente sync() qui — bloccava ogni richiesta HTTP per il tempo del
    # rescan+upload completo su R2 (visto causare timeout 504 dal tunnel).
    # La sincronizzazione va lanciata a parte: `python3 sync_r2.py` oppure
    # tramite l'endpoint /sync qui sotto.
    sections = []
    if ITER_DIR.exists():
        tags = sorted(
            [p for p in ITER_DIR.iterdir() if p.is_dir()],
            key=lambda p: (
                int(m.group(1)) if (m := re.match(r"v(\d+)_", p.name)) else 999
            ),
        )
        for tag_dir in tags:
            images = sorted(
                p for p in tag_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            if not images:
                continue
            cards = "\n".join(
                f'''<figure class="card">
                    <a href="{PUBLIC_URL}/iterazioni/{tag_dir.name}/{img.name}" data-lightbox="{tag_dir.name}">
                        <img src="{PUBLIC_URL}/thumbs/{tag_dir.name}/{img.stem}.jpg" loading="lazy" alt="{img.name}">
                    </a>
                    <figcaption>{img.name}</figcaption>
                </figure>'''
                for img in images
            )
            sections.append(f'''
                <section class="iteration">
                    <h2>{_human_label(tag_dir.name)}</h2>
                    <div class="grid">{cards}</div>
                </section>
            ''')

    root_images = sorted(
        p for p in ITER_DIR.glob("*.png") if p.is_file()
    ) if ITER_DIR.exists() else []
    if root_images:
        cards = "\n".join(
            f'''<figure class="card">
                <a href="{PUBLIC_URL}/iterazioni/{img.name}" data-lightbox="confronti">
                    <img src="{PUBLIC_URL}/thumbs/{img.stem}.jpg" loading="lazy" alt="{img.name}">
                </a>
                <figcaption>{img.name}</figcaption>
            </figure>'''
            for img in root_images
        )
        sections.insert(0, f'''
            <section class="iteration">
                <h2>Confronti diretti</h2>
                <div class="grid">{cards}</div>
            </section>
        ''')

    body = "\n".join(sections) if sections else "<p>Nessuna iterazione trovata ancora.</p>"

    return f'''<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review rimozione gruccia</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 2rem;
    background: #0e0e10; color: #e8e8ea;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #9a9aa2; margin-bottom: 2rem; font-size: 0.9rem; }}
  section.iteration {{ margin-bottom: 3rem; }}
  h2 {{
    font-size: 1.05rem; border-bottom: 1px solid #2a2a2e; padding-bottom: 0.5rem;
    margin-bottom: 1rem; color: #f0f0f2;
  }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1rem;
  }}
  .card {{
    margin: 0; background: #18181b; border: 1px solid #2a2a2e; border-radius: 8px;
    overflow: hidden;
  }}
  .card img {{
    width: 100%; height: 220px; object-fit: contain; background: #000;
    display: block; cursor: zoom-in;
  }}
  .card figcaption {{
    font-size: 0.72rem; padding: 0.4rem 0.6rem; color: #9a9aa2;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  #lightbox-overlay {{
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.92);
    z-index: 999; align-items: center; justify-content: center;
    overflow: hidden; touch-action: none;
  }}
  #lightbox-overlay.open {{ display: flex; }}
  #lightbox-img {{
    max-width: 94vw; max-height: 94vh; object-fit: contain;
    box-shadow: 0 0 40px rgba(0,0,0,0.6);
    transform-origin: 0 0; cursor: zoom-in;
    /* niente transizione durante il trascinamento: il pan deve seguire il
       mouse senza ritardo, altrimenti via desktop remoto sembra rotto */
    will-change: transform;
  }}
  #lightbox-img.zoomed {{ cursor: grab; max-width: none; max-height: none; }}
  #lightbox-img.dragging {{ cursor: grabbing; }}
  #zoom-hud {{
    position: fixed; bottom: 1rem; left: 50%; transform: translateX(-50%);
    background: rgba(20,20,24,0.9); border: 1px solid #3a3a40; border-radius: 8px;
    padding: 0.5rem 0.9rem; display: flex; gap: 0.75rem; align-items: center;
    font-size: 0.8rem; color: #d8d8dc; z-index: 1000; user-select: none;
  }}
  #zoom-hud button {{
    background: #2a2a30; color: #e8e8ea; border: 1px solid #43434a;
    border-radius: 5px; padding: 0.25rem 0.6rem; cursor: pointer;
    font-size: 0.85rem; line-height: 1.2;
  }}
  #zoom-hud button:hover {{ background: #35353d; }}
  #zoom-level {{ min-width: 3.6rem; text-align: center; font-variant-numeric: tabular-nums; }}
  #zoom-hint {{ color: #8a8a92; font-size: 0.72rem; }}
  #lightbox-close {{
    position: fixed; top: 1rem; right: 1.25rem; z-index: 1000;
    background: rgba(20,20,24,0.9); color: #e8e8ea; border: 1px solid #3a3a40;
    border-radius: 6px; padding: 0.3rem 0.7rem; cursor: pointer; font-size: 1rem;
  }}
</style>
</head>
<body>
  <h1>Review rimozione gruccia — Scout</h1>
  <div class="subtitle">Generato automaticamente da output/iterazioni/. Clicca un'immagine per ingrandirla.</div>
  {body}

  <div id="lightbox-overlay">
    <img id="lightbox-img" src="">
    <button id="lightbox-close" title="Chiudi (Esc)">✕</button>
    <div id="zoom-hud">
      <button data-zoom="out" title="Riduci (-)">−</button>
      <span id="zoom-level">adatta</span>
      <button data-zoom="in" title="Ingrandisci (+)">+</button>
      <button data-zoom="100" title="Pixel reali (1)">1:1</button>
      <button data-zoom="fit" title="Adatta a schermo (0)">adatta</button>
      <span id="zoom-hint">rotella = zoom · trascina = sposta · doppio clic = 1:1</span>
    </div>
  </div>
  <script>
    // Lightbox con zoom/pan tipo e-commerce. Serve perche' i confronti
    // affiancati sono larghi ~2700px: adattati a schermo si perde meta' dei
    // pixel, e via desktop remoto la differenza si nota.
    const overlay = document.getElementById('lightbox-overlay');
    const lbImg = document.getElementById('lightbox-img');
    const hudLevel = document.getElementById('zoom-level');

    let scale = 0;        // 0 = "adatta a schermo" (nessuna trasformazione)
    let tx = 0, ty = 0;   // traslazione in px
    let natW = 0, natH = 0;
    const MIN = 0.1, MAX = 8;

    function fitScale() {{
      if (!natW) return 1;
      return Math.min(window.innerWidth * 0.94 / natW,
                      window.innerHeight * 0.94 / natH);
    }}

    function render() {{
      if (scale === 0) {{
        lbImg.classList.remove('zoomed');
        lbImg.style.transform = '';
        hudLevel.textContent = 'adatta';
        return;
      }}
      lbImg.classList.add('zoomed');
      lbImg.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
      hudLevel.textContent = Math.round(scale * 100) + '%';
    }}

    // Passa da "adattata" a trasformazione esplicita mantenendo la posizione
    // visiva: senza questo, il primo zoom fa saltare l'immagine.
    function materialize() {{
      if (scale !== 0) return;
      const r = lbImg.getBoundingClientRect();
      scale = fitScale();
      tx = r.left; ty = r.top;
    }}

    function zoomAt(cx, cy, factor) {{
      materialize();
      const ns = Math.min(MAX, Math.max(MIN, scale * factor));
      // il punto sotto il cursore resta fermo
      tx = cx - (cx - tx) * (ns / scale);
      ty = cy - (cy - ty) * (ns / scale);
      scale = ns;
      render();
    }}

    function setZoom(target) {{
      materialize();
      const cx = window.innerWidth / 2, cy = window.innerHeight / 2;
      zoomAt(cx, cy, target / scale);
    }}

    function reset() {{ scale = 0; tx = ty = 0; render(); }}

    function open(href) {{
      lbImg.src = href;
      overlay.classList.add('open');
      reset();
    }}

    function close() {{
      overlay.classList.remove('open');
      lbImg.src = '';
      reset();
    }}

    lbImg.addEventListener('load', () => {{
      natW = lbImg.naturalWidth; natH = lbImg.naturalHeight;
    }});

    document.querySelectorAll('a[data-lightbox]').forEach(a => {{
      a.addEventListener('click', e => {{ e.preventDefault(); open(a.getAttribute('href')); }});
    }});

    // Chiude solo cliccando lo sfondo: sull'immagine il clic serve al pan.
    overlay.addEventListener('click', e => {{ if (e.target === overlay) close(); }});
    document.getElementById('lightbox-close').addEventListener('click', close);

    overlay.addEventListener('wheel', e => {{
      e.preventDefault();
      zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.15 : 1 / 1.15);
    }}, {{ passive: false }});

    lbImg.addEventListener('dblclick', e => {{
      e.preventDefault();
      if (scale !== 0 && Math.abs(scale - 1) < 0.01) reset();
      else {{ materialize(); zoomAt(e.clientX, e.clientY, 1 / scale); }}
    }});

    let dragging = false, sx = 0, sy = 0;
    lbImg.addEventListener('pointerdown', e => {{
      if (scale === 0) return;
      dragging = true; sx = e.clientX - tx; sy = e.clientY - ty;
      lbImg.classList.add('dragging');
      lbImg.setPointerCapture(e.pointerId);
      e.preventDefault();
    }});
    lbImg.addEventListener('pointermove', e => {{
      if (!dragging) return;
      tx = e.clientX - sx; ty = e.clientY - sy;
      render();
    }});
    lbImg.addEventListener('pointerup', e => {{
      dragging = false; lbImg.classList.remove('dragging');
      lbImg.releasePointerCapture(e.pointerId);
    }});

    document.querySelectorAll('#zoom-hud button').forEach(b => {{
      b.addEventListener('click', e => {{
        e.stopPropagation();
        const k = b.dataset.zoom;
        if (k === 'in') zoomAt(innerWidth / 2, innerHeight / 2, 1.3);
        else if (k === 'out') zoomAt(innerWidth / 2, innerHeight / 2, 1 / 1.3);
        else if (k === '100') setZoom(1);
        else reset();
      }});
    }});

    document.addEventListener('keydown', e => {{
      if (!overlay.classList.contains('open')) return;
      if (e.key === 'Escape') close();
      else if (e.key === '+' || e.key === '=') zoomAt(innerWidth / 2, innerHeight / 2, 1.3);
      else if (e.key === '-') zoomAt(innerWidth / 2, innerHeight / 2, 1 / 1.3);
      else if (e.key === '1') setZoom(1);
      else if (e.key === '0') reset();
    }});
  </script>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
async def index():
    return build_page()


@app.post("/sync")
async def trigger_sync():
    import asyncio

    uploaded = await asyncio.to_thread(sync)
    return {"uploaded": len(uploaded)}


if __name__ == "__main__":
    print("Apri: http://127.0.0.1:8090")
    uvicorn.run(app, host="127.0.0.1", port=8090)
