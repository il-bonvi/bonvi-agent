# Piano modello ad ampie vedute

Obiettivo:
- Replicare il tuo occhio su effort e sprint.
- Scoprire pattern nuovi utili, senza perdere coerenza con i gold.

## Ciclo di lavoro (sempre uguale)

1. Aggiungi nuove sessioni in data/archive.
2. Lancia AGGIORNA_MODELLO.bat.
3. Apri SOLO_REPORT.bat.
4. Leggi in ordine:
   - models/training_summary.md
   - models/patterns_found.md
   - models/training_history.csv
5. Prendi 1-3 decisioni operative e annotale nel log esperimenti.

## Regola decisionale semplice

Se migliora davvero:
- effort_f1 sale e sprint_f1 sale
- effort/sprint delta_mae scendono
- trend stabile per almeno 2-3 run

Se peggiora:
- controlla prima coerenza dei gold
- verifica se i nuovi dati sono molto diversi dai precedenti
- non cambiare troppe cose insieme

## Strategia ad ampie vedute

1. Base umana (sempre):
- avg power
- rolling_30s centered
- rolling_60s centered

2. Base algoritmo default (gia inclusa):
- effort_config
- sprint_config
- feature trim/extend

3. Pattern scoperti (da validare):
- usa models/patterns_found.md
- prendi solo i pattern ripetuti in piu run
- annota quali pattern diventano regole esplicite

## Cosa fare ogni settimana

1. Aggiungi nuove triplette buone.
2. Esegui retrain.
3. Confronta con settimana precedente.
4. Segna una sola modifica prioritaria.

## Errore da evitare

- Ottimizzare su una singola gara/sessione.
- Il target e la robustezza, non il risultato perfetto su un caso solo.
