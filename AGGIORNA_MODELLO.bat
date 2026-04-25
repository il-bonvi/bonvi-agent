@echo off
setlocal

cd /d "%~dp0"

echo ============================================
echo BONVI AGENT - AGGIORNAMENTO MODELLO
echo ============================================

if not exist ".venv\Scripts\python.exe" goto :noenv

set PY=.venv\Scripts\python.exe

if not exist "data\archive" (
  echo ERRORE: cartella data\archive non trovata.
  goto :fail
)

echo [1/3] Validazione archivio...
"%PY%" scripts\validate_archive.py --archive data\archive
if errorlevel 1 goto :fail

echo [2/3] Build dataset training...
"%PY%" scripts\build_training_dataset.py --archive data\archive --out-dir data\processed
if errorlevel 1 goto :fail

echo [3/3] Training modelli...
"%PY%" scripts\train_models.py --effort-csv data\processed\effort_training.csv --sprint-csv data\processed\sprint_training.csv --model-root models
if errorlevel 1 goto :fail

echo.
echo COMPLETATO: modello aggiornato con successo.
echo Report: data\processed\archive_validation_report.json
echo Report: data\processed\dataset_build_report.json
echo Metriche: models\training_metrics.json
echo.
pause
exit /b 0

:noenv
echo ERRORE: ambiente virtuale non trovato.
echo Esegui una sola volta:
echo   python -m venv .venv
echo   .venv\Scripts\activate
echo   pip install -r requirements.txt
echo.
pause
exit /b 1

:fail
echo.
echo BLOCCATO: controlla errori mostrati sopra.
echo.
pause
exit /b 1
