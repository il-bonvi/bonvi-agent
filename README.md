# Bonvi Effort Detection Agent

Repository dedicata al training del modello su file che fornisci tu.

Non genera i JSON: usa le triplette gia prodotte dal tuo software esterno.

## Dove inserire i file

Metti tutto sotto `data/archive`.

Regola: una sessione per cartella, con questi 4 file:

- `esempio.fit`
- `esempio_default.json`
- `esempio_gold.json`
- `effort_modifications.json` (ignorato automaticamente)

Puoi organizzare gare e allenamenti in sottocartelle come preferisci, ad esempio:

```text
data/archive/
  gare/
    2026-03-10_Criterium_Roma/
      roma.fit
      roma_default.json
      roma_gold.json
      effort_modifications.json
  allenamenti/
    2026-03-12_Soglia_Colli/
      colli.fit
      colli_default.json
      colli_gold.json
      effort_modifications.json
```

Il sistema cerca le sessioni in modo ricorsivo, quindi questa struttura e supportata.

## Cosa fa questa repo

1. Valida struttura e JSON delle sessioni.
2. Costruisce dataset training per effort e sprint confrontando default vs gold.
3. Allena i modelli (classifier + regressor) con validazione leave-one-session-out.
4. Salva modelli e metriche in `models/`.

## Setup rapido

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Flusso operativo

### 1) Validazione archivio

```bash
python scripts/validate_archive.py --archive data/archive
```

Output principale:

- `data/processed/archive_validation_report.json`

Se ci sono errori di layout (stem mismatch, file mancanti) o payload invalidi, il comando esce con codice errore.

### 2) Build dataset training

```bash
python scripts/build_training_dataset.py --archive data/archive --out-dir data/processed
```

Output:

- `data/processed/effort_training.csv`
- `data/processed/sprint_training.csv`
- `data/processed/dataset_build_report.json`

Note:

- `effort_modifications.json` viene ignorato.
- Le label sono create confrontando segmenti default vs gold con IoU.

### 3) Training modelli

```bash
python scripts/train_models.py --effort-csv data/processed/effort_training.csv --sprint-csv data/processed/sprint_training.csv --model-root models
```

Output:

- `models/classifier/effort_keep_xgb.joblib`
- `models/regressor/effort_start_delta_xgb.joblib` (se dati positivi sufficienti)
- `models/regressor/effort_end_delta_xgb.joblib` (se dati positivi sufficienti)
- `models/classifier/sprint_keep_xgb.joblib`
- `models/regressor/sprint_start_delta_xgb.joblib` (se dati positivi sufficienti)
- `models/regressor/sprint_end_delta_xgb.joblib` (se dati positivi sufficienti)
- `models/training_metrics.json`

## Regole dati importanti

- Stesso stem tra `.fit`, `_default.json`, `_gold.json` nella stessa cartella sessione.
- Stessa `session_id` tra default e gold della stessa sessione.
- Campi richiesti in `session_info`: `session_id`, `filename`, `cp`, `weight`, `activity_type`.
- `activity_type` ammessi: `training`, `freeride`, `road`, `criterium`, `ITT`, `TTT`.

## Struttura codice

- `src/archive.py`: scoperta sessioni e lettura JSON.
- `src/dataset_builder.py`: costruzione dataset effort/sprint da triplette.
- `src/model_training.py`: training modelli e metriche LOSO.
- `src/loader.py`: lettura FIT.
- `src/rolling.py`: rolling power.
- `src/schema.py`: validazione schema JSON.

## Cosa e stato rimosso

Sono stati eliminati i componenti di generazione JSON e annotazione interna, per allinearsi al tuo processo reale:

- niente detector interno per creare `_default.json`
- niente editor annotation Streamlit

Questa repo ora e focalizzata solo su ingestione triplette fornite da te + training modello.
