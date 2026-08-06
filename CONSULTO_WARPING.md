# Consulto tecnico: raddrizzare la forma di capi d'abbigliamento fotografati su gruccia

Ciao. Sto lavorando a una pipeline di post-produzione per foto e-commerce e sono
bloccato su un problema di computer vision. Ti chiedo un parere tecnico, anche
critico: se pensi che l'impostazione sia sbagliata in partenza, dimmelo.

## Contesto

Foto di capi d'abbigliamento (brand Scout) appesi a una gruccia, su fondo bianco
uniforme, per un catalogo e-commerce. Risoluzioni alte: da 2000x3000 fino a
4375x6671 px. Pipeline in Python (OpenCV, numpy, PyTorch), GPU RTX 3060 12GB.

La pipeline ha due compiti:

1. **Rimuovere la gruccia** — RISOLTO. SAM 3 con prompt testuale produce la
   maschera, LaMa fa l'inpainting. Funziona bene. (Nota che potrebbe esserti
   utile: abbiamo provato 7 motori di inpainting; tutti quelli "intelligenti"
   — ZITS, MIGAN, FcF, PowerPaint, SD 1.5 — RICOSTRUISCONO la gruccia invece di
   cancellarla, perché la struttura più evidente dentro la maschera *è* la
   gruccia. LaMa funziona proprio perché non ragiona: estende lo sfondo.)

2. **Correggere la forma del capo** — È QUI CHE SONO BLOCCATO.

## Il problema preciso

Il capo appeso si deforma: le clip della gruccia tirano il tessuto, il capo pende
storto. Il retoucher umano in Photoshop usa il **Liquify** per correggere.

Esempio concreto e rappresentativo: una **gonna lunga a balze** (bianca, tessuto
leggero, 6 balze orizzontali sovrapposte). Le due clip della gruccia afferrano la
vita e la tirano su in due punti, quindi **la vita risulta storta rispetto al
resto della gonna**, che invece cade dritta. Va raddrizzata la vita senza toccare
il resto.

**Cosa NON è il problema**: non è una rotazione globale del capo. Ruotare tutta
l'immagine non serve — ruoterebbe anche le parti già corrette. La deformazione è
**locale**.

## Cosa ho già costruito e cosa funziona

**Il motore di deformazione: FUNZIONA.** Un warp locale con campo di
spostamento gaussiano applicato via `cv2.remap`:

- input: lista di trascinamenti `{x0,y0 -> x1,y1, raggio, forza}`
- effetto che sfuma con gaussiana attorno al punto di presa
- zone "congelate" (rettangoli che non si devono muovere, es. l'orlo)
- campo calcolato su griglia rada (passo 8-16px) e poi ingrandito con
  INTER_CUBIC, perché il campo è liscio
- ~0.5 s su un capo 3561x6601, tutto su CPU
- **nessun artefatto visibile**, nessun pixel generato: si ridistribuisce solo
  quello che c'è (scelta voluta, coerente con la lezione dell'inpainting)

Ho verificato che funziona: indicando a mano due punti sulla vita storta, la
gonna si raddrizza in modo pulito e l'orlo resta immobile (0 pixel modificati
sotto la zona congelata).

## Cosa NON funziona: il rilevamento automatico di COSA raddrizzare

Ho fatto due tentativi, entrambi falliti, per motivi diversi.

### Tentativo 1: profilo esterno della sagoma — FALLITO

Idea: estrarre la sagoma (alpha o soglia sul bianco), prendere il profilo
superiore, adattarci una retta, misurarne l'inclinazione.

Perché ha fallito: **il profilo superiore di un capo non è orizzontale per
costruzione.** Le spalle di una camicia o di una maglietta scendono in diagonale
da entrambi i lati (forma a V). Adattarci una retta produce un numero privo di
significato — le due diagonali opposte si compensano e danno ~0° anche su un capo
palesemente storto. Le dispersioni residue erano di 100-120px su capi alti
3000-3500px: la retta non descriveva niente.

Sintomo iniziale ancora peggiore: misurando sulla foto grezza ottenevo angoli di
**+63°**, perché la sagoma includeva ancora gancio e staffa metallica e stavo
misurando la ferramenta. (Risolto scartando le righe superiori troppo strette,
ma il metodo resta sbagliato nel merito.)

### Tentativo 2: linea interna via gradiente — FALLITO IN MODO PIÙ INTERESSANTE

Idea (correggendo l'errore precedente): non il contorno esterno, ma una **linea
interna** al capo — il bordo della fascia in vita, una cucitura. Quelle nel capo
reale SONO orizzontali, quindi la loro inclinazione nella foto è esattamente il
difetto.

Implementazione: Sobel verticale (risponde ai bordi orizzontali), mascherato sul
capo, mediato per righe dentro la fascia alta (4%-35% dell'altezza del capo).
Poi confronto separato tra metà sinistra e metà destra: il picco cade a quote
diverse, e la differenza è l'inclinazione. Più un punteggio di "nitidezza" del
picco rispetto alla mediana, come confidenza, e un rifiuto se |angolo| > 12°.

Risultati su 12 capi (6 modelli, 2 scatti ciascuno):
- 9 "corretti" con angoli tra -4.4° e +4.0°
- 3 rifiutati: 2 camicie a quadri ("nessuna linea netta", nitidezza 0.27-0.28)
  e un vestito ("inclinazione implausibile -55°")

**Ma ecco il punto che mi ha fatto fermare.** Sulla gonna a balze:
- sullo **scatto grezzo** (gruccia ancora presente): trova -4.39°, applica, e il
  risultato sembra buono a occhio
- sullo **stesso capo dopo la rimozione della gruccia**: trova **+47.3°** e
  rifiuta. Ha agganciato y=954 a sinistra e y=2478 a destra — 1524px di scarto,
  cioè **due balze diverse**, non la stessa linea.

Il che mi fa sospettare che il -4.39° sullo scatto grezzo fosse in buona parte
**fortuna**: il gradiente più forte nella fascia alta era probabilmente il bordo
della gruccia stessa, non la vita. Un riferimento accidentale.

Il problema di fondo: una gonna a balze ha **6+ linee orizzontali candidate**, e
niente nel gradiente distingue "la vita" da "la terza balza". Su un tessuto a
quadri il segnale è ovunque; su un tessuto uniforme non c'è.

## Le domande

1. **Il rilevamento automatico della linea da raddrizzare è recuperabile?**
   Idee che ho considerato ma non provato:
   - vincolare i due lati a quote simili (scarto max 5% dell'altezza) — eviterebbe
     l'aggancio a balze diverse, ma non garantisce di trovare quella giusta
   - Hough sulle linee, raggruppando per orientamento invece di prendere il max
   - cercare la linea più vicina al punto in cui le clip afferravano il capo:
     **quella posizione la conosco già**, perché è la maschera di SAM 3 che ho
     usato per rimuovere la gruccia. La deformazione è causata dalle clip, quindi
     la zona da correggere è sotto di esse. Questa mi sembra la pista migliore ma
     voglio un parere prima di investirci.
   - un modello di landmark detection per abbigliamento (DeepFashion2 ha 294
     landmark per categoria) — sovradimensionato? praticabile?

2. **C'è un approccio completamente diverso che non sto vedendo?** Per esempio
   ragionare sulla simmetria del capo rispetto al proprio asse verticale, o sulla
   griglia della tessitura, o qualcosa dal mondo del garment/cloth modeling.

3. **Oppure la risposta onesta è che va lasciato semi-manuale?** Cioè: l'operatore
   fa due clic sulla vita e il motore (che funziona) raddrizza. Su un catalogo di
   centinaia di capi sarebbe comunque molto più veloce di Photoshop. Se questa è
   la risposta giusta, preferisco saperlo adesso invece di inseguire un automatismo
   che non regge.

Un vincolo di metodo che mi sono dato: **niente generativo per la deformazione.**
Il capo fotografato è il prodotto reale che il cliente riceverà, quindi non posso
inventare tessuto o pattern. Solo ridistribuzione di pixel esistenti.

Grazie del parere.
