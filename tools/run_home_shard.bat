@echo off
setlocal
pushd "%~dp0.."
set SEED_START=%~1
set GAMES=%~2
set DEALER=%~3
set TARGET=%~4
set OUT=%~5
if "%SEED_START%"=="" set SEED_START=100000
if "%GAMES%"=="" set GAMES=100
if "%DEALER%"=="" set DEALER=0
if "%TARGET%"=="" set TARGET=0
if "%OUT%"=="" set OUT=outputs\home_shard_%SEED_START%_%GAMES%_d%DEALER%_t%TARGET%.json
if not exist outputs mkdir outputs
python tools\run_swap_shard.py --seed-start %SEED_START% --games %GAMES% --dealer %DEALER% --target %TARGET% --out "%OUT%"
set STATUS=%ERRORLEVEL%
popd
if not "%STATUS%"=="0" pause
exit /b %STATUS%
