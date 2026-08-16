# Test esplorativo: Qwen Image Edit (GGUF) su VPS Vast.ai

Non fa parte della pipeline di produzione (LaMa/PowerPaint/patch_fill, vedi
README.md). È un test per capire se Qwen Image Edit, usato con un crop di
contesto stretto e compositing manuale, ricostruisce meglio di LaMa le zone
semantiche grandi (colletto, spalle) dove oggi il fill appiattisce il
tessuto.

Gira in un **venv separato** (`venv-qwen`) dal resto del progetto, perché
richiede `diffusers`/`transformers` molto più recenti di quelli usati da
IOPaint nell'ambiente principale — non toccano quindi l'ambiente esistente.

---

## 1. Noleggiare l'istanza su Vast.ai

- **GPU**: RTX 4090 24GB (target primario)
- **Disco**: **100 GB** consigliati (60 GB è il minimo stretto). Vedi la
  stima dettagliata più sotto.
- **Immagine**: una immagine `pytorch` ufficiale (es. `pytorch/pytorch`
  con CUDA 12.1 già pronta) — evita di dover installare driver/toolkit a
  mano. `setup_vast.sh`/`setup_qwen.sh` installano comunque torch da soli
  se manca, ma partire da un'immagine PyTorch fa risparmiare tempo.

Al termine del noleggio, Vast fornisce host e porta SSH (es.
`ssh -p 40123 root@123.45.67.89`). Sostituisci `<HOST>` e `<PORTA>` con
quei valori in tutti i comandi sotto.

---

## 2. Portare il codice sull'istanza

Due modi equivalenti, entrambi già pronti nel progetto:

### A — clone diretto sull'istanza (più semplice)

```bash
ssh -p <PORTA> root@<HOST>
git clone https://github.com/daveinme/removal-script.git
cd removal-script
```

### B — sync da locale (porta anche i capi di test)

Dal tuo pc, dentro `prototipo-gruccia/`:

```bash
./deploy-remote.sh sync root@<HOST> -p <PORTA>
```

Copia codice + immagini di test (`postproduzione/IMMAGINI STYLESHOOT/`),
escludendo venv/output/modelli (si rigenerano sull'istanza).

---

## 3. Ambiente principale (SAM 3 + LaMa) — solo se serve generare la mask

Se la maschera della gruccia non è già stata generata altrove, serve prima
l'ambiente standard per la detection:

```bash
./deploy-remote.sh setup root@<HOST> -p <PORTA>
```

oppure, già sull'istanza:

```bash
bash setup_vast.sh
```

~15 minuti: crea `venv/`, scarica SAM 3 (~3.3 GB), precarica i modelli
IOPaint. Poi si genera la mask con la GUI (`./venv/bin/python gui/app.py`,
vedi tunnel al punto 6) oppure con `detect_segment_test.py`.

Se la mask esiste già (es. portata da un run precedente), questo passo si
può saltare.

---

## 4. Ambiente Qwen GGUF

Da locale:

```bash
./deploy-remote.sh qwen-setup root@<HOST> -p <PORTA>
```

oppure, già sull'istanza:

```bash
bash setup_qwen.sh Q5_K_M
```

Lo script stampa **timestamp e tempi reali** ad ogni fase (niente stime a
priori) e produce output di questo tipo:

```
=== [14:32:01] (+0s totali) Setup ambiente Qwen Image Edit GGUF — quant Q5_K_M ===
  disco: /dev/sda ... 98G ...

=== [14:32:03] (+2s totali) Creo venv-qwen ===
=== [14:34:40] (+159s totali) Installo diffusers/transformers recenti + accelerate + gguf ===
  torch 2.5.1 | diffusers 0.31.0
  GPU: NVIDIA GeForce RTX 4090 — 25.8 GB

=== [14:35:10] (+189s totali) Scarico i pesi (transformer GGUF Q5_K_M + pipeline base) ===
[14:35:10] --> transformer GGUF Q5_K_M (~14 GB per Q5_K_M)
[14:38:47] transformer scaricato in 217s
[14:38:47] --> pipeline base Qwen-Image-Edit-2509 (text_encoder+VAE, ~17-18 GB)
[14:43:02] pipeline base scaricata in 255s

=== [14:43:02] (+661s totali) Spazio occupato dalla cache HuggingFace ===
  32G  /root/.cache/huggingface/hub
  disco: /dev/sda ... 58G ...

=== [14:43:02] (+661s totali) Fatto in 661s totali ===
```

I tempi di download variano con la velocità di rete dell'istanza scelta —
per questo lo script mostra sempre l'orario e il delta, non una stima
fissa.

### Quant disponibili

Passabile come argomento (`bash setup_qwen.sh <QUANT>`):

| Quant | Dimensione transformer | Note |
|---|---|---|
| Q4_K_M | ~11 GB | più margine VRAM, possibile perdita di dettaglio fine (cuciture) |
| **Q5_K_M** | **~14 GB** | **default consigliato**, compromesso qualità/VRAM |
| Q8_0 | ~19 GB | quasi lossless, rischia di stringere troppo insieme al text encoder |

---

## 5. Lanciare il test

Serve una mask già generata (`output/<nome_capo>_mask.png`, o dentro
`output/iterazioni/<tag>/`, stessa convenzione di `inpaint_test.py`).

```bash
./venv-qwen/bin/python src/qwen_test.py <nome_capo> \
    --quant Q5_K_M \
    --steps 30 \
    --garment "felpa"
```

Output in `output/` (o `output/iterazioni/<tag>/` se passi `--tag`):

- `<nome_capo>_qwen_Q5_K_M_<ora>.png` — risultato finale compositato
  (originale + solo l'area della mask sostituita, con feathering)
- `<nome_capo>_qwen_Q5_K_M_<ora>_crop_raw.png` — output grezzo del
  modello sul crop, prima del compositing (utile per capire se il
  problema è la generazione o il blending)

Argomenti utili:

- `--margin`: pixel di contesto extra intorno alla mask dati in pasto al
  modello (default 220px) — più contesto aiuta a dedurre pieghe/cuciture,
  ma allarga il crop e quindi il tempo di generazione
- `--tag`: per tenere separati esperimenti diversi, come già fa
  `inpaint_test.py`

---

## 6. Vedere i risultati: due modi

### A — tunnel SSH classico (richiede sessione aperta sul tuo pc)

```bash
./deploy-remote.sh tunnel root@<HOST> -p <PORTA>
```

poi apri **http://localhost:8095** nel browser.

### B — tunnel Cloudflare pubblico (nessuna sessione locale da tenere aperta)

```bash
./deploy-remote.sh cf-tunnel root@<HOST> -p <PORTA>
```

Installa `cloudflared` sull'istanza (una tantum), avvia la GUI e stampa un
URL pubblico temporaneo tipo `https://xxxx-yyyy.trycloudflare.com` —
raggiungibile da qualunque browser, niente account Cloudflare richiesto.
Il link scade quando chiudi la sessione (Ctrl+C chiude sia tunnel che GUI
sull'istanza). Non serve per `qwen_test.py` in sé (è un test a riga di
comando), ma comodo per confrontare gli output nella GUI insieme agli
altri motori.

Per vedere solo i file PNG generati da `qwen_test.py` senza passare dalla
GUI, più semplice riportarli in locale:

```bash
./deploy-remote.sh pull root@<HOST> -p <PORTA>
```

Scarica `output/` in `output/remoto/` sul tuo pc.

---

## Stima spazio disco

| Componente | Dimensione |
|---|---|
| Transformer GGUF Q5_K_M | ~14 GB |
| Pipeline base (text_encoder bf16 + VAE) | ~17-18 GB |
| venv-qwen (torch + diffusers + transformers) | ~8-10 GB |
| SAM 3 + venv principale (se serve generare la mask) | ~10-15 GB |
| Margine cache HF + immagini test + output | ~10 GB |
| **Totale consigliato** | **~80-100 GB** |

Se scarichi solo Qwen (mask già pronta, niente ambiente principale), **60
GB** bastano stretti.

---

## Nota tecnica: perché il crop e non l'immagine intera

`QwenImageEditPipeline` (diffusers) è **edit-by-prompt sull'intera
immagine**, come Kontext — non esiste una mask nativa che limiti l'area
modificata (a differenza di `QwenImageInpaintPipeline`, non testata qui).
Per questo `qwen_test.py`:

1. ritaglia un crop di contesto attorno alla mask (non l'immagine intera)
2. genera con un prompt che descrive esplicitamente cosa ricostruire
3. compone il risultato **solo dentro la mask dilatata**, con feathering
   — mai `result = generated`

Stesso principio di compositing già usato da `fill_v3()` in `gui/app.py`
per gli altri motori: il modello generativo propone, il compositing
decide cosa entra davvero nel risultato finale.

## Nota tecnica: perché il text encoder va scaricato durante l'inferenza

Il text encoder di Qwen-Image-Edit (Qwen2.5-VL, ~16 GB in bf16) serve solo
per codificare il prompt all'inizio, non durante i passi di diffusione.
`qwen_test.py` usa `pipe.enable_model_cpu_offload()`: il text encoder resta
in RAM di sistema e viene spostato in VRAM solo quando serve, così durante
la generazione vera e propria in VRAM restano solo transformer GGUF (~14
GB con Q5_K_M) + VAE, entro i 24 GB della 4090.
