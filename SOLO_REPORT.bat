@echo off
setlocal

cd /d "%~dp0"

echo ============================================
echo BONVI AGENT - APERTURA REPORT
echo ============================================

set R1=data\processed\archive_validation_report.json
set R2=data\processed\dataset_build_report.json
set R3=models\training_metrics.json
set R4=models\training_history.csv
set R5=models\training_summary.md

if not exist "%R1%" echo Manca: %R1%
if not exist "%R2%" echo Manca: %R2%
if not exist "%R3%" echo Manca: %R3%
if not exist "%R4%" echo Manca: %R4%
if not exist "%R5%" echo Manca: %R5%

if exist "%R1%" start "" "%R1%"
if exist "%R2%" start "" "%R2%"
if exist "%R3%" start "" "%R3%"
if exist "%R4%" start "" "%R4%"
if exist "%R5%" start "" "%R5%"

echo.
echo Aperto tutto quello che era disponibile.
echo.
pause
exit /b 0
