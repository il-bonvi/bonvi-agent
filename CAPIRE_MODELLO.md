# Capire cosa fa il modello

Questo file spiega, in modo pratico, cosa succede quando lanci AGGIORNA_MODELLO.bat.

## Flusso in 3 step

1. Validazione archivio
- Controlla che ogni cartella sessione abbia i file giusti.
- Controlla che i JSON siano validi.

2. Build dataset
- Confronta default vs gold.
- Crea righe di training per effort e sprint.

Nota importante sulle medie usate dal modello:
- avg 30s e avg 60s sono CENTERED (non causali).
- avg30: usa contesto intorno al punto (circa 15s prima + 15s dopo).
- avg60: usa contesto intorno al punto (circa 30s prima + 30s dopo).
- Questa scelta allinea le feature al tuo criterio di annotazione nel file gold.

3. Training
- Allena i modelli.
- Salva metriche, storico e modelli finali.

## File creati e cosa contengono

### data/processed/archive_validation_report.json
A cosa serve:
- Ti dice se i dati in input sono puliti.

Cosa contiene:
- archive: cartella analizzata
- valid_session_folders: quante sessioni valide
- layout_errors: errori nomi/struttura
- payload_errors: errori contenuto JSON

Interpretazione rapida:
- Se layout_errors e payload_errors sono vuoti, puoi andare avanti.

### data/processed/dataset_build_report.json
A cosa serve:
- Ti dice quanto dataset hai generato.

Cosa contiene:
- sessions_used
- effort_rows
- sprint_rows
- warnings

Interpretazione rapida:
- Se effort_rows e sprint_rows crescono nel tempo, stai aggiungendo dati utili.

### data/processed/effort_training.csv
A cosa serve:
- Dataset per addestrare la parte effort.

Cosa contiene (colonne principali):
- session_id, stem, folder, activity_type
- cp, weight
- feature di potenza normalizzate (avg_power_ratio, avg_30_ratio, avg_60_ratio, ecc.)
- keep_label: 1 = tenere, 0 = scartare
- start_delta_sec, end_delta_sec: correzioni boundary verso il gold

### data/processed/sprint_training.csv
A cosa serve:
- Dataset per addestrare la parte sprint.

Cosa contiene (colonne principali):
- session_id, stem, activity_type
- cp, weight
- duration_sec, avg_power_ratio
- keep_label
- start_delta_sec, end_delta_sec

### models/training_metrics.json
A cosa serve:
- Report tecnico completo dell'ultimo training.

Cosa contiene:
- effort_classifier_loso: accuracy, precision, recall, f1
- effort_regressor_loso: start_delta_mae, end_delta_mae
- sprint_classifier_loso: accuracy, precision, recall, f1
- sprint_regressor_loso: start_delta_mae, end_delta_mae
- saved_files: elenco modelli salvati

Interpretazione rapida:
- Classification: accuracy/f1 piu alte = meglio.
- Regressione: mae piu bassa = meglio.

### models/training_history.csv
A cosa serve:
- Storico di tutti i training nel tempo.

Cosa contiene:
- una riga per ogni run (timestamp)
- dimensione dataset (sessions_count, effort_rows, sprint_rows)
- metriche effort/sprint

Interpretazione rapida:
- sessions_count deve salire.
- effort_f1 e sprint_f1 idealmente salgono.
- effort/sprint delta_mae idealmente scendono.

### models/training_summary.md
A cosa serve:
- Versione leggibile al volo dell'ultimo training.

Cosa contiene:
- sessions, effort rows, sprint rows
- metriche principali effort/sprint in forma breve

Consiglio:
- Apri prima questo file, poi il JSON se vuoi dettaglio tecnico.

### models/classifier/*.joblib
A cosa serve:
- Modelli di classificazione (tenere/scartare).

File tipici:
- effort_keep_xgb.joblib
- sprint_keep_xgb.joblib

Cosa c'e dentro:
- modello XGBoost addestrato
- lista colonne usate in training

### models/regressor/*.joblib
A cosa serve:
- Modelli di correzione boundary (start/end).

File tipici:
- effort_start_delta_xgb.joblib
- effort_end_delta_xgb.joblib
- sprint_start_delta_xgb.joblib
- sprint_end_delta_xgb.joblib

Cosa c'e dentro:
- modello XGBoost regressione
- lista colonne usate in training

## Come capire se il modello migliora davvero

Controlla ogni settimana questi 5 numeri:

1. sessions_count (training_history.csv)
- Deve aumentare.

2. effort_f1 (training_history.csv)
- Deve tendere a salire.

3. sprint_f1 (training_history.csv)
- Deve tendere a salire.

4. effort_start_delta_mae + effort_end_delta_mae
- Devono tendere a scendere.

5. sprint_start_delta_mae + sprint_end_delta_mae
- Devono tendere a scendere.

Regola pratica:
- Dopo 20-30 triplette buone, le metriche iniziano a stabilizzarsi.
- Se aggiungi dati incoerenti, il modello puo peggiorare.

## Routine consigliata

1. Aggiungi nuove sessioni in data/archive.
2. Lancia AGGIORNA_MODELLO.bat.
3. Apri SOLO_REPORT.bat.
4. Guarda prima training_summary.md.
5. Controlla trend in training_history.csv.
