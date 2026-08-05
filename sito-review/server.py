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
    z-index: 999; align-items: center; justify-content: center; cursor: zoom-out;
  }}
  #lightbox-overlay.open {{ display: flex; }}
  #lightbox-overlay img {{
    max-width: 94vw; max-height: 94vh; object-fit: contain;
    box-shadow: 0 0 40px rgba(0,0,0,0.6);
  }}
</style>
</head>
<body>
  <h1>Review rimozione gruccia — Scout</h1>
  <div class="subtitle">Generato automaticamente da output/iterazioni/. Clicca un'immagine per ingrandirla.</div>
  {body}

  <div id="lightbox-overlay"><img id="lightbox-img" src=""></div>
  <script>
    const overlay = document.getElementById('lightbox-overlay');
    const lbImg = document.getElementById('lightbox-img');
    document.querySelectorAll('a[data-lightbox]').forEach(a => {{
      a.addEventListener('click', e => {{
        e.preventDefault();
        lbImg.src = a.getAttribute('href');
        overlay.classList.add('open');
      }});
    }});
    overlay.addEventListener('click', () => {{
      overlay.classList.remove('open');
      lbImg.src = '';
    }});
    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape') {{ overlay.classList.remove('open'); lbImg.src=''; }}
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
