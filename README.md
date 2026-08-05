# Rimozione gruccia — Scout

Pipeline di post-produzione per foto prodotto: rimuove la gruccia (barra,
gancio, clip) dalle immagini dei capi, a piena risoluzione, senza reinventare
la texture del tessuto.

**Detection**: SAM 3 con prompt testuale → **Fill**: LaMa (o altri motori a
confronto) → GUI di controllo capo per capo.

---

## Avvio rapido su istanza GPU (Vast, RunPod…)

```bash
git clone https://github.com/daveinme/removal-script.git
cd removal-script
bash setup_vast.sh          # ambiente + SAM 3 + modelli IOPaint (~15 min)
./venv/bin/python gui/app.py
```

Dal proprio pc, per usare la GUI nel browser:

```bash
ssh -p <PORTA> -L 8095:localhost:8095 root@<HOST>
```

poi aprire **http://localhost:8095**. Le immagini si caricano con drag&drop.
Non servono API pubbliche né webhook: la GUI è una web app locale raggiunta
via tunnel SSH.

### Pesi ZITS (opzionale, download manuale)

Da Google Drive, in `modelli/zits_repo/ckpt/`:

- [LSM-HAWP](https://drive.google.com/drive/folders/1yg4Nc20D34sON0Ni_IOezjJCFHXKGWUW) → `best_lsm_hawp.pth`
- [Pesi ZITS](https://drive.google.com/drive/folders/1Dg_6ZCAi0U3HzrYgXwr9nSaOLnPsf9n-) → `best_transformer_places2.pth`, `StructureUpsampling.pth`, `InpaintingModel_best_gen.pth`

L'ultimo va copiato anche in `ckpt/zits_places2_hr/`.

---

## Stato dei test (aggiornato 2026-08-05)

Sette motori di fill provati su capi reali. **LaMa resta il migliore.**

| Motore | Esito | Note |
|---|---|---|
| **LaMa** | ✅ **migliore** | Estende lo sfondo invece di ricostruire: è ciò che serve |
| ZITS | ❌ | Lascia un *fantasma a tratteggio* della gruccia |
| MIGAN | ❌ | Stesso fantasma + frammenti colorati inventati |
| FcF | ❌ | Stesso fantasma, benché sia famiglia Fourier come LaMa |
| MAT | ⚠️ OOM su 12 GB | **da testare su 24 GB** |
| LDM | ⚠️ OOM (chiede 40 GB) | **da testare su 24 GB** |
| SD 1.5 | ❌ per rimuovere | Inventa: ha prodotto una gruccia *zebrata*. ✅ per **sostituire** |
| RMBG-2.0 | ❌ | Considera la gruccia parte del soggetto |
| Texture patching | ❌ | Non trova mai una sorgente abbastanza grande |

### Il pattern emerso

I modelli **che "capiscono la struttura" ricostruiscono la gruccia** invece di
cancellarla — la struttura più evidente dentro la maschera *è* la gruccia.
LaMa funziona proprio perché non ragiona sulla struttura.

FcF lo dimostra: stessa architettura Fourier di LaMa, ma addestramento diverso
→ stesso fallimento di ZITS. **Non è l'architettura, è l'addestramento.**

### Da testare sulla 4090 24GB

1. **PowerPaint** (già nel menu, task `object-remove`) — l'unico addestrato
   *specificamente* per la rimozione oggetti, non un prompt generico.
   Il più promettente: è un diffusion ma guidato a cancellare, quindi non
   dovrebbe inventare come SD 1.5
2. **MAT** e **LDM** — mai visti lavorare davvero, solo OOM
3. **FLUX.2 Klein 4B** — Apache 2.0, piena risoluzione, ~13 GB.
   Richiede `transformers>=5` che rompe groundingdino: usare un **venv
   separato** (`venv-flux`), passando le maschere come PNG già generate.

---

## Casi ancora irrisolti

1. **Ombre delle clip sul tessuto** — SAM 3 segmenta la clip ma non l'ombra
   che proietta: resta un alone scuro. Implementato `add_shadows()` in
   `gui/app.py`, **da tarare** (nel primo test non ha agito).
2. **Gruccia dietro un dettaglio strutturale** (colletto di una maglia) —
   il fill riempie con lo sfondo invece che col tessuto che continua dietro.
3. **Morso delle clip su tessuto a fantasia** — il tessuto era fisicamente
   occluso: va ricostruito. Nessun modello lo fa in modo affidabile, e
   nessuna GPU cambia la fisica del problema. Candidato al ritocco manuale.

---

## La GUI

`gui/app.py` (FastAPI, porta 8095) + `gui/index.html`.

- **Drag & drop**, immagini solo in memoria: al refresh riparte pulita
- **Detector**: SAM 3 (prompt testuale) o GroundingDINO+SAM2 (storico)
- **Prompt**: `clothes hanger, hook, clip`.
  ⚠️ **Non aggiungere `tape`**: confonde il nastro con le etichette gialle
  del capo e le cancella
- **Confidenza** (default 0.25): quali oggetti passano il filtro
- **Dilatazione** (default 15px): impostazione v3, quella che danneggiava
  meno il capo
- **Ricostruzione**: LaMa / MAT / MIGAN / FcF / ZITS / LDM / SD 1.5 / nessuna
- **Metrica chiave**: *px del capo alterati* — più bassa è meglio.
  È la misura che ha rivelato la regressione delle vecchie iterazioni v13-v15

### Nota sulla VRAM

SAM 3 occupa ~9 GB e viene scaricato dopo la detection (`free_vram()`).
I modelli IOPaint vengono tenuti uno alla volta. Su 24 GB il vincolo si
allenta parecchio.

---

## Licenze dei modelli (verificate sui repository)

| Modello | Licenza | Uso commerciale |
|---|---|---|
| LaMa | Apache 2.0 | ✅ |
| SAM 3 | licenza SAM | ✅ |
| ZITS | Apache 2.0 | ✅ |
| PowerPaint v2 | Apache 2.0 | ✅ |
| FLUX.2 Klein 4B | Apache 2.0 | ✅ |
| **EdgeConnect** | CC BY-NC 4.0 | ❌ **no** |
| **MAT** | "research only" | ❌ **no** |

I pesi SAM 3 vengono dal mirror `1038lab/sam3` perché `facebook/sam3` è
gated. Per un uso commerciale conviene chiedere l'accesso ufficiale.
