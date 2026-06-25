# Knowledge-Constrained Full Waveform Inversion via SIREN/PINN

> Studio sull'effetto dell'incorporazione di colonne di velocità note (well log) come vincoli spaziali all'interno di un framework di inversione sismica basato su Physics-Informed Neural Networks (PINN) con architettura SIREN.

**Istituzione:** Image and Sound Processing Lab (ISPL) – Politecnico di Milano  
**Linguaggio:** Python 3 (100%)  
**Framework deep learning:** PyTorch

---

## Indice

1. [Panoramica del progetto](#1-panoramica-del-progetto)
2. [Concetti chiave](#2-concetti-chiave)
3. [Struttura della repository](#3-struttura-della-repository)
4. [Descrizione dei file](#4-descrizione-dei-file)
5. [Requisiti e installazione](#5-requisiti-e-installazione)
6. [Pipeline completa: passo per passo](#6-pipeline-completa-passo-per-passo)
7. [Argomenti da riga di comando](#7-argomenti-da-riga-di-comando)
8. [Strategia dei vincoli: Case A vs Case B](#8-strategia-dei-vincoli-case-a-vs-case-b)
9. [Scheduling dell'alpha](#9-scheduling-dellalpha)
10. [Output e metriche](#10-output-e-metriche)
11. [Struttura dati attesa](#11-struttura-dati-attesa)
12. [Risultati principali](#12-risultati-principali)
13. [Note tecniche avanzate](#13-note-tecniche-avanzate)

---

## 1. Panoramica del progetto

La **Full Waveform Inversion (FWI)** è una tecnica di imaging sismico ad alta risoluzione che ricostruisce il modello di velocità del sottosuolo minimizzando la differenza tra dati sismici osservati e sintetici. È notoriamente sensibile alla qualità del modello iniziale e tende a convergere verso minimi locali in assenza di informazioni geologiche a priori.

Questo progetto propone un'estensione **knowledge-constrained** di un framework FWI basato su SIREN/PINN, in cui colonne di velocità note (ad esempio da log di pozzo) vengono incorporate nella fase di inversione come **vincoli spaziali hard o soft**.

Sono investigate due strategie di vincolo:

- **Case A** – Iniezione diretta delle colonne note nel modello iniziale (vincolo hard, via pre-training condizionato).
- **Case B** – Termine di penalizzazione soft aggiunto alla loss function (implementato in `fwi_method2_sceltaColonneEMetriche.py`).

Esperimenti su tre modelli di benchmark — **Marmousi**, **Overthrust** e **BP2004** — dimostrano che il **Case B** supera sistematicamente il baseline non vincolato e il Case A.

---

## 2. Concetti chiave

### SIREN (Sinusoidal Representation Network)
Rete neurale con attivazioni sinusoidali (Sitzmann et al., 2020) usata come **rappresentazione implicita** del modello di velocità 2D. La rete mappa coordinate spaziali normalizzate `(x, z) ∈ [-1,1]²` a un valore scalare di velocità `vp(x,z)`. Il suo bias induttivo liscio agisce come regolarizzatore implicito per la FWI, prevenendo artefatti ad alta frequenza.

### PINN (Physics-Informed Neural Network)
La rete viene ottimizzata non solo sul mismatch dei dati sismici, ma anche sulla fisica del problema (equazione d'onda acustica). Il forward modeling è implementato tramite differenze finite con schema leapfrog del 2° ordine nel tempo e Laplaciano del **4° ordine** nello spazio (ridotta dispersione numerica).

### PML (Perfectly Matched Layer)
Strato assorbente ai bordi del dominio di calcolo per eliminare riflessioni artificiali dai bordi. Il dominio fisico viene espanso di `nbl` campioni per lato, e i coefficienti PML vengono generati automaticamente.

### Column penalty loss
Il vincolo soft del Case B aggiunge alla loss totale un termine MSE tra le colonne di velocità predette dalla rete e i valori reali noti:

```
total_loss = α(t) · data_loss + (1 - α(t)) · col_loss
```

dove `α(t)` è schedulato dinamicamente durante l'addestramento.

---

## 3. Struttura della repository

```
.
├── pinn_utils.py                          # Libreria core: fisica, SIREN, utilities
├── pretrain_siren.py                      # Step 1: pre-addestramento SIREN sul modello iniziale
├── forward.py                             # Step 2: forward modeling (generazione shot gather)
├── fwi_method2_sceltaColonneEMetriche.py  # Step 3: inversione FWI con vincoli a colonne
├── VelocityModelsPNG/                     # Immagini di riferimento dei modelli di velocità
└── README.md                              # Questo file
```

**Struttura dati attesa (non inclusa nella repo, da preparare):**

```
data/
├── v_models/          # Modelli di velocità (.npz)
│   ├── marmousi_paper_sp15.npz
│   ├── marmousi_paper_sm10_sp15.npz   # versione smoothed per pre-training
│   └── ...
├── siren/             # Pesi SIREN pre-addestrati (.pth)
│   └── marmousi_paper_sm10_sp15.pth
├── shots/             # Shot gather sintetici (.npz) prodotti da forward.py
│   └── marmousi_paper_sp15.npz
└── output/            # Risultati FWI (creato automaticamente)
    └── fwi_method2_*/
        ├── fwi_best_model.pth
        ├── fwi_final_model.pth
        ├── fwi_results.npz
        ├── metrics.txt
        ├── rmse_mae_summary.png
        ├── png/               # Snapshot dell'inversione (ogni N epoch)
        └── abs_diff/          # Mappe di errore assoluto 2D
```

---

## 4. Descrizione dei file

### `pinn_utils.py` — Libreria core (557 righe)

Contiene tutti i building block riutilizzati dagli altri script:

| Funzione/Classe | Descrizione |
|---|---|
| `set_gpu(id)` | Seleziona la GPU (auto: quella con meno memoria usata) |
| `check_cfl(dt_s, dh_km, vp_max, strict)` | Verifica la condizione di stabilità CFL; se `strict=True` lancia `ValueError` |
| `generate_pml_coefficients_2d(shape, N)` | Genera i coefficienti di damping PML 2D |
| `absorbing_boundaries(nx, ny, nb, u)` | Alternativa esponenziale al PML |
| `laplace(u, h, dev)` | Laplaciano 2D al **4° ordine** FD (meno dispersione rispetto al 2° ordine) |
| `step(u_pre, u_now, c, dt, h, b)` | Avanza l'onda di un time step (schema leapfrog + PML) |
| `forward(wave, c, b, src_list, ...)` | Forward modeling batched su GPU: propaga tutti gli shot in parallelo |
| `bandpass_shots(shots, f_low, f_high, dt_s)` | Filtro Butterworth per FWI multi-scala (frequency continuation) |
| `SineLayer` | Layer lineare + attivazione `sin(ω₀ · x)` con inizializzazione specifica SIREN |
| `Siren` | Rete SIREN completa: genera la griglia di coordinate, supporta pesi pre-addestrati |
| `step_elastic`, `forward_elastic` | Stub per equazione d'onda elastica (non usato nel workflow principale) |

---

### `pretrain_siren.py` — Step 1: Pre-training (202 righe)

Addestra la rete SIREN a rappresentare un **modello di velocità iniziale smooth** (tipicamente una versione a bassa frequenza spaziale del modello vero). Questo fornisce un punto di partenza migliore per la FWI rispetto all'inizializzazione random.

**Dettagli di training:**
- Loss: MSE tra output SIREN e modello target
- Ottimizzatore: AdamW (`lr=1e-4`, `weight_decay=1e-5`)
- Scheduler: `ReduceLROnPlateau` (halving se loss stagnante per 50 epoch)
- Gradient clipping: `max_norm=1.0`
- `omega_0 = 10` (conservativo, evita overfitting alle alte frequenze spaziali)
- Salva il **miglior checkpoint** (non solo l'ultimo)

**Output:** `data/siren/<nome_modello>.pth`

---

### `forward.py` — Step 2: Forward modeling (272 righe)

Genera i **dati sismici sintetici osservati** su cui verrà addestrata la FWI. Simula la propagazione dell'onda acustica nel modello di velocità vero per tutti gli shot.

**Flusso:**
1. Carica il modello di velocità vero (`.npz`)
2. Verifica la stabilità CFL (errore bloccante se violata)
3. Costruisce la geometria sorgenti/ricevitori
4. Genera il wavelet Ricker con frequenza dominante `f0`
5. Genera i coefficienti PML
6. Esegue il forward modeling in batch su GPU
7. Salva i gather in `data/shots/<nome>.npz`

**Output NPZ contiene:** `d_obs_list`, `src_coordinates`, `rec_coordinates`, `wave`, `domain_pad`, `pmlc`, `dt`, `tn`, `nbl`, `spacing`, `domain`

---

### `fwi_method2_sceltaColonneEMetriche.py` — Step 3: FWI con vincoli (661 righe)

Script principale dell'inversione. Implementa il **Case B**: loss con penalizzazione soft sulle colonne note, con scheduling dinamico del peso `α`.

**Flusso dell'inversione:**
1. Carica shot gather osservati e metadati
2. Seleziona le colonne di velocità note (modalità `spaced` o `random`)
3. Inizializza la rete SIREN dai pesi pre-addestrati
4. Loop FWI:
   - Calcola `α(t)` secondo lo schedule scelto
   - Seleziona un mini-batch di shot
   - Forward pass SIREN → velocità `vp(x,z)` (clampata in `[1.5, 4.5]` km/s)
   - Padding PML + forward modeling → sismogrammi sintetici
   - Calcola `data_loss` e `col_loss`
   - `total_loss = α · data_loss + (1-α) · col_loss`
   - Backward + gradient clipping + AdamW step
5. Ogni `epochs_per_plot` epoch: salva metriche, snapshot PNG, mappa errore assoluto
6. Al termine: salva `fwi_results.npz`, `metrics.txt`, summary plots

---

## 5. Requisiti e installazione

### Dipendenze Python

```bash
pip install torch numpy matplotlib scipy tqdm GPUtil
```

Versioni testate: Python ≥ 3.9, PyTorch ≥ 2.0, CUDA ≥ 11.8 (raccomandato per GPU).

### Clone della repository

```bash
git clone https://github.com/AndyPaglia/Seismic-Inversion-Study-using-PINN-SIREN-Columns-Knowlege-Considerations.git
cd Seismic-Inversion-Study-using-PINN-SIREN-Columns-Knowlege-Considerations
```

### Preparazione dei dati

I modelli di velocità di benchmark devono essere scaricati separatamente e convertiti in formato `.npz` con chiavi `vp` (array 2D in km/s, shape `[Nx, Nz]`) e `spacing` (spaziatura in metri).

Modelli usati nel paper:
- **Marmousi** – classico modello 2D con strutture complesse
- **Overthrust** – modello con thrust fault
- **BP2004** – modello di difficoltà geologica massima

---

## 6. Pipeline completa: passo per passo

### Step 1 — Pre-addestrare il SIREN

```bash
python pretrain_siren.py \
    --vp_model_path ./data/v_models/marmousi_paper_sm10_sp15.npz \
    --epochs 1000 \
    --plot
```

Il modello smooth (`sm10`) è la versione a bassa frequenza spaziale del modello vero. Il SIREN impara a rappresentarlo prima che inizi la FWI vera.

### Step 2 — Generare i dati osservati

```bash
python forward.py \
    --vp_model_path ./data/v_models/marmousi_paper_sp15.npz \
    --src_spacing 300 \
    --rec_spacing 15 \
    --rec_depth 0 \
    --src_depth 30 \
    --f0 0.008 \
    --tn 1900 \
    --dt 1.9 \
    --nbl 100 \
    --batch_size 1000 \
    --plot
```

> **Attenzione:** il parametro `--f0` è in kHz (0.008 kHz = 8 Hz).

### Step 3 — Eseguire la FWI (Case B, configurazione consigliata)

```bash
python fwi_method2_sceltaColonneEMetriche.py \
    --obs_data_path ./data/shots/marmousi_paper_sp15.npz \
    --siren_path ./data/siren/marmousi_paper_sm10_sp15.pth \
    --true_vp_path ./data/v_models/marmousi_paper_sp15.npz \
    --out_dir ./data/output/fwi_m2_20col_spaced_linear \
    --fwi_iterations 10000 \
    --shots_per_epoch 5 \
    --n_known_cols 20 \
    --col_selection_mode spaced \
    --alpha_schedule linear \
    --alpha_start 0.1 \
    --alpha_end 0.9 \
    --shot_selection_policy random \
    --plot
```

---

## 7. Argomenti da riga di comando

### `pretrain_siren.py`

| Argomento | Default | Descrizione |
|---|---|---|
| `--vp_model_path` | `./data/v_models/marmousi_paper_sm10_sp15.npz` | Modello di velocità smooth per il pre-training |
| `--epochs` | `1000` | Numero di epoche |
| `--plot` | `False` | Mostra plot diagnostici durante il training |

---

### `forward.py`

| Argomento | Default | Descrizione |
|---|---|---|
| `--vp_model_path` | `./data/v_models/marmousi_paper_sp15.npz` | Modello di velocità vero |
| `--src_spacing` | `300` | Spaziatura tra sorgenti [m] |
| `--rec_spacing` | `15` | Spaziatura tra ricevitori [m] |
| `--rec_depth` | `0` | Profondità ricevitori [m] |
| `--src_depth` | `30` | Profondità sorgenti [m] |
| `--f0` | `0.008` | Frequenza dominante wavelet Ricker [kHz] |
| `--tn` | `1900` | Tempo di registrazione finale [ms] |
| `--dt` | `1.9` | Time step [ms] |
| `--nbl` | `100` | Numero di layer PML per lato |
| `--batch_size` | `1000` | Shot per batch GPU |
| `--plot` | `False` | Mostra plot diagnostici |

---

### `fwi_method2_sceltaColonneEMetriche.py`

#### Input/Output

| Argomento | Default | Descrizione |
|---|---|---|
| `--obs_data_path` | `./data/shots/marmousi_paper_sp15.npz` | Shot gather osservati (da `forward.py`) |
| `--siren_path` | `./data/siren/marmousi_paper_sm10_sp15.pth` | Pesi SIREN pre-addestrati |
| `--true_vp_path` | `./data/v_models/marmousi_paper_sp15.npz` | Modello vero (per metriche) |
| `--out_dir` | `./data/output/fwi_method2_...` | Cartella di output |

#### Parametri di training

| Argomento | Default | Descrizione |
|---|---|---|
| `--fwi_iterations` | `10000` | Numero totale di epoche FWI |
| `--shots_per_epoch` | `5` | Shot per mini-batch |
| `--shot_selection_policy` | `random` | Politica di selezione shot: `random`, `sequential`, `spaced` |
| `--all_shots` | `False` | Usa tutti gli shot ogni epoca (suddivisi in mini-batch) |
| `--multiscale` | `False` | FWI multi-scala: bandpass progressivo (frequency continuation) |
| `--plot` | `False` | Mostra plot durante il training |
| `--debug` | `False` | Disabilita salvataggio su disco (solo test rapidi) |

#### Selezione colonne vincolate

| Argomento | Default | Descrizione |
|---|---|---|
| `--n_known_cols` | `None` | Numero di colonne note da usare come vincolo |
| `--col_selection_mode` | `spaced` | Modalità: `spaced` (distribuzione uniforme) o `random` |
| `--col_seed` | `42` | Seed per selezione random riproducibile |
| `--known_col_spacing` | `50` | Usato se `n_known_cols` è None: spacing fisso tra colonne |
| `--col_range_start` | `0` | Indice colonna iniziale del range ammissibile |
| `--col_range_end` | `None` | Indice colonna finale del range (None = fino alla fine) |
| `--col_range_start_2` | `None` | Secondo range di colonne (per pool non contiguo) |
| `--col_range_end_2` | `None` | Fine secondo range |

#### Scheduling alpha

| Argomento | Default | Descrizione |
|---|---|---|
| `--alpha_schedule` | `fixed` | Schedule: `fixed`, `linear`, `sigmoid`, `cosine` |
| `--alpha_start` | `0.5` | Valore iniziale di α (peso `data_loss`) |
| `--alpha_end` | `0.9` | Valore finale di α |
| `--alpha_pivot` | `0.2` | Per `sigmoid`: punto di inflessione normalizzato (0-1) |
| `--alpha_steepness` | `10.0` | Per `sigmoid`: ripidità della transizione |
| `--convergence_threshold` | `1e-4` | Soglia di loss per logging convergenza (non ferma il training) |

---

## 8. Strategia dei vincoli: Case A vs Case B

### Case A — Iniezione diretta (hard constraint)

Le colonne note vengono incorporate nel modello iniziale prima dell'inversione. Il SIREN viene pre-addestrato su un modello che già include i valori esatti delle colonne di pozzo. Questo è un vincolo **hard**: la rete parte già "ancorata" ai valori noti, ma non esiste meccanismo che la impedisca di deviare durante la FWI.

### Case B — Penalizzazione soft (implementata in questo repo)

Durante ogni step di ottimizzazione FWI, la loss totale include un termine aggiuntivo che penalizza la deviazione dalle colonne note:

```
col_loss = MSE( vp_pred[:, col_idx] / col_scale,
                vp_true[:, col_idx] / col_scale )

total_loss = α · data_loss + (1 - α) · col_loss
```

Entrambi i termini sono **normalizzati** per la scala del segnale (rispettivamente `obs_scale` e `col_scale`), rendendoli comparabili indipendentemente dalla magnitudine dei dati.

**Perché il Case B vince:** il termine di penalizzazione guida continuamente l'ottimizzazione verso valori fisicamente plausibili in posizioni note, senza bloccare il gradiente del dato che porta informazione sulle zone non vincolate.

---

## 9. Scheduling dell'alpha

Il parametro `α(t)` bilancia durante il training il peso relativo tra il termine di dato (sismico) e quello di vincolo (colonne):

| Schedule | Comportamento |
|---|---|
| `fixed` | α costante = `alpha_start` per tutto il training |
| `linear` | α aumenta linearmente da `alpha_start` a `alpha_end` |
| `sigmoid` | Transizione sigmoide; controllata da `alpha_pivot` e `alpha_steepness` |
| `cosine` | Transizione coseno (smooth, senza bruschi salti) |

**Configurazione vincente nei benchmark:** `linear` con `alpha_start=0.1`, `alpha_end=0.9`. Questo fa sì che nelle prime epoche il vincolo di colonna domini (guida verso un modello plausibile), mentre nelle epoche finali il dato sismico abbia il peso maggiore (affina i dettagli).

---

## 10. Output e metriche

### File salvati da `fwi_method2_sceltaColonneEMetriche.py`

| File | Descrizione |
|---|---|
| `fwi_best_model.pth` | Pesi SIREN all'epoca con loss minima |
| `fwi_final_model.pth` | Pesi SIREN all'ultima epoca |
| `fwi_results.npz` | Modello invertito, array di loss, metriche per checkpoint, history α |
| `metrics.txt` | Sommario leggibile: configurazione, timing, convergenza, metriche per checkpoint |
| `rmse_mae_summary.png` | Plot finale RMSE e MAE vs modello vero |
| `png/epoch*.png` | Snapshot a 5 pannelli ogni N epoche: modello invertito, loss totale, loss raw, RMSE/MAE, α schedule |
| `abs_diff/diff_epoch*.png` | Mappa dell'errore assoluto `|vp_pred - vp_true|` per checkpoint |

### Metriche tracciate

- **`total_loss`** – Loss pesata totale (metrica di ottimizzazione)
- **`data_loss` (raw)** – MSE normalizzato tra sismogrammi sintetici e osservati
- **`col_loss` (raw)** – MSE normalizzato tra colonne predette e colonne vere
- **`RMSE`** – Root Mean Square Error rispetto al modello vero (km/s)
- **`MAE`** – Mean Absolute Error rispetto al modello vero (km/s)
- **`α(t)`** – Storia dello scheduling

Le metriche **primarie** per valutare la qualità dell'inversione sono `data_loss` e `RMSE` (o `MAE`) rispetto al modello vero.

---

## 11. Struttura dati attesa

### Formato modello di velocità (`.npz`)

```python
{
    "vp":      np.ndarray,  # shape (Nx, Nz), valori in km/s
    "spacing": np.ndarray,  # shape (2,), [dh_x, dh_z] in metri
}
```

> **Nota:** all'interno del codice viene applicata la trasposizione `.T` per ottenere la convenzione `(Nz, Nx)` usata internamente.

### Formato shot gather (`.npz`) — output di `forward.py`

```python
{
    "d_obs_list":    np.ndarray,  # shape (nshots, nt, nrec)
    "src_coordinates": np.ndarray,  # shape (nshots, 2), [x, z] in km
    "rec_coordinates": np.ndarray,  # shape (nrec, 2)
    "wave":          np.ndarray,  # wavelet Ricker, shape (nt,)
    "domain_pad":    np.ndarray,  # (Nz_pad, Nx_pad)
    "domain":        np.ndarray,  # (Nz, Nx) fisico
    "pmlc":          np.ndarray,  # coefficienti PML, shape (Nz_pad, Nx_pad)
    "dt":            float,       # time step in ms
    "tn":            float,       # tempo finale in ms
    "nbl":           int,         # numero layer PML
    "spacing":       np.ndarray,  # spaziatura in m
}
```

---

## 12. Risultati principali

I seguenti risultati provengono dagli esperimenti descritti nel paper allegato:

| Configurazione | MAE relativo vs baseline | Convergenza |
|---|---|---|
| Baseline (no vincoli) | 0% (riferimento) | — |
| Case A (20 col. spaced) | −15% circa | Simile |
| Case B (20 col. spaced, linear α) | **−40% circa** (su BP2004) | **~2× più veloce** |
| Case B (20 col. random, linear α) | −20% circa | Instabile |

**Conclusioni principali:**
- La distribuzione **uniforme (spaced)** delle colonne è critica: il posizionamento random degrada accuratezza e stabilità.
- Lo schedule **lineare** (da α=0.1 a α=0.9) è superiore a sigmoid e cosine, soprattutto sui modelli geologicamente complessi.
- Aumentare il numero di colonne vincolate migliora monotonicamente i risultati fino a circa 20 colonne, dopo di che il guadagno marginale si riduce.

---

## 13. Note tecniche avanzate

### Stabilità numerica CFL

La condizione di Courant–Friedrichs–Lewy deve essere soddisfatta:

```
dt ≤ dh / (√2 · vp_max)
```

Il codice verifica automaticamente questa condizione in `check_cfl()`. Se violata, viene lanciato un `ValueError` bloccante (comportamento `strict=True` di default).

### Laplaciano al 4° ordine

Rispetto al classico stencil al 2° ordine, i coefficienti `[-1/12, 4/3, -5/2, 4/3, -1/12]` riducono significativamente la dispersione numerica, consentendo l'uso di frequenze più alte per lo stesso grid spacing.

### Clamp della velocità

Durante la FWI, la velocità predetta dalla rete viene clampata a `[1.5, 4.5] km/s` per garantire fisicità:

```python
vp = torch.clamp(vp, min=1.5, max=4.5)
```

### GPU auto-selection

`set_gpu(-1)` seleziona automaticamente la GPU con il minor utilizzo di memoria tramite `GPUtil`, utile in ambienti multi-GPU condivisi (es. cluster ISPL).

### Multi-scala (frequency continuation)

Con `--multiscale`, il training viene suddiviso in 3 stage con bandpass crescente sui dati:
- Stage 1: 1–5 Hz
- Stage 2: 1–10 Hz  
- Stage 3: banda completa

Questo approccio riduce il rischio di convergenza a minimi locali nelle prime iterazioni.

### Normalizzazione delle loss

Entrambi i termini di loss sono normalizzati per la scala del rispettivo segnale:

```python
data_loss = MSE(syn / obs_scale, obs_batch / obs_scale)
col_loss  = MSE(vp_cols / col_scale, known_cols / col_scale)
```

Questo garantisce che il bilanciamento tramite `α` sia interpretabile indipendentemente dalla magnitudine dei segnali coinvolti.

---

## Citazione

Se usi questo codice nel tuo lavoro, cita:

> Paglialunga, A. et al. *Knowledge-Constrained Full Waveform Inversion via SIREN/PINN: Incorporating Well Log Spatial Constraints into Neural Seismic Imaging*. ISPL, Politecnico di Milano.

---

## Licenza

Vedi la repository GitHub per i dettagli sulla licenza.
