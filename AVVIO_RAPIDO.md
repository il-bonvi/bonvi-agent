# Avvio rapido (super semplice)

1. Una sola volta, prepara ambiente:
   - python -m venv .venv
   - .venv\\Scripts\\activate
   - pip install -r requirements.txt

2. Ogni volta che aggiungi nuove sessioni:
   - metti cartelle in data/archive (una sessione per cartella)
   - ogni cartella sessione deve avere:
     - esempio.fit
     - esempio_default.json
     - esempio_gold.json
     - effort_modifications.json (ignorato)

3. Doppio click su AGGIORNA_MODELLO.bat

Il .bat fa tutto in automatico:
1. valida archivio
2. costruisce dataset
3. allena modello

Report finali:
- data/processed/archive_validation_report.json
- data/processed/dataset_build_report.json
- models/training_metrics.json
- models/training_history.csv (storico automatico: 1 riga a ogni training)
- models/training_summary.md (versione leggibile)

Come leggere lo storico in 10 secondi:
- sessions_count deve crescere nel tempo
- effort_f1 e sprint_f1 idealmente salgono
- effort_start_delta_mae / effort_end_delta_mae idealmente scendono

Se il CSV non ti piace, apri prima training_summary.md.

Se il .bat si blocca, di solito e per:
- stem diverso tra fit, _default.json, _gold.json
- session_id diversa tra default e gold

Quando hai circa 20-30 triplette buone, puoi iniziare test seri di rilevamento automatico
effort/sprint secondo il tuo stile.

Opzionale (periodico): puoi pianificare AGGIORNA_MODELLO.bat con Utilita di pianificazione
di Windows (es. ogni domenica sera).

Per aprire al volo tutti i report dopo il training: doppio click su SOLO_REPORT.bat
