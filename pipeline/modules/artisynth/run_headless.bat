@ECHO OFF
REM ==========================================================================
REM  run_headless.bat START END [NPROC] [OUT] [POOL]   (Step 3, Windows bare metal)
REM
REM  Headless (-noGui) static mesh export from the Step-2 pool via
REM  export_static_impl.py. NPROC>1 splits [START,END) into NPROC ranges
REM  (aligned to 1000-index shards) and launches that many parallel workers
REM  (minimized, logs to <OUT>\log). Disjoint 1000-aligned ranges => no shard
REM  race. Re-run the SAME args to resume: finished shards are skipped.
REM
REM  Configuration (env vars, or edit the defaults below):
REM    ARTISYNTH_HOME  artisynth_core checkout (compiled: classes\ + lib\*)
REM    TONGUE_MODEL    ArtiSynth model class
REM  Defaults for OUT / POOL sit right below; the pipeline-relative POOL default
REM  matches config.json (outputs\pool.txt).
REM
REM  Calls java directly instead of bin\artisynth.bat, because that launcher
REM  sends -D flags to the Main class, not the JVM, so the system properties
REM  would never arrive.
REM
REM  Examples:
REM    run_headless.bat 0 295157 4
REM    run_headless.bat 150000 295157 4 D:\static_300k
REM  RAM ~ NPROC*6GB, NPROC cores. START must be a multiple of 1000 for clean
REM  boundaries.
REM ==========================================================================
setlocal enabledelayedexpansion
if not defined ARTISYNTH_HOME set ARTISYNTH_HOME=C:\Users\d11\artisynth\artisynth_core
if not defined TONGUE_MODEL set TONGUE_MODEL=artisynth.models.tongue3d.StableFemMuscleTongueDemo
set AH=%ARTISYNTH_HOME%
set SCRIPT=%~dp0export_headless.py
set IMPL=%~dp0export_impl.py
set OUT=D:\static_300k
set POOL=%~dp0..\..\outputs\pool.txt

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage
set START=%~1
set END=%~2
set NPROC=%~3
if "%NPROC%"=="" set NPROC=1
if not "%~4"=="" set OUT=%~4
if not "%~5"=="" set POOL=%~5

if not exist "%POOL%" (
  echo pool file not found: %POOL%
  echo run step2_muscle_sampling\sample_pool.py first, or pass POOL as 5th arg
  exit /B 1
)

if not "%NPROC%"=="1" goto :dispatch

echo [worker] HEADLESS range %START%..%END%  out=%OUT%  pool=%POOL%
if defined WLOG goto :wlog
java -Xmx6G -Dstatic.start=%START% -Dstatic.end=%END% -Dstatic.pool="%POOL%" -Dstatic.out="%OUT%" -Dstatic.impl="%IMPL%" -Dstatic.model=%TONGUE_MODEL% -cp "%AH%\classes;%AH%\lib\*" artisynth.core.driver.Main -noGui -model %TONGUE_MODEL% -script "%SCRIPT%"
exit /B %ERRORLEVEL%
:wlog
java -Xmx6G -Dstatic.start=%START% -Dstatic.end=%END% -Dstatic.pool="%POOL%" -Dstatic.out="%OUT%" -Dstatic.impl="%IMPL%" -Dstatic.model=%TONGUE_MODEL% -cp "%AH%\classes;%AH%\lib\*" artisynth.core.driver.Main -noGui -model %TONGUE_MODEL% -script "%SCRIPT%" > "%WLOG%" 2>&1
exit /B %ERRORLEVEL%

:dispatch
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%OUT%\log" mkdir "%OUT%\log"
set /a SPAN=END-START
set /a CH=(SPAN + NPROC - 1)/NPROC
set /a CH=((CH + 999)/1000)*1000
set /a LAST=NPROC-1
echo Dispatching up to %NPROC% HEADLESS workers over %START%..%END%  chunk %CH% (1000-aligned)  out=%OUT%
for /L %%i in (0,1,%LAST%) do (
  set /a LO=START + %%i*CH
  set /a HI=!LO! + CH
  if !HI! GTR %END% set HI=%END%
  if !LO! LSS %END% (
    set "WLOG=%OUT%\log\w%%i_!LO!_!HI!.log"
    echo   worker %%i : range !LO!..!HI!   log !WLOG!
    start "tongue w%%i" /MIN cmd /c call "%~f0" !LO! !HI! 1 "%OUT%" "%POOL%"
    timeout /t 3 /nobreak >nul
  )
)
set "WLOG="
echo Workers launched.
exit /B 0

:usage
echo usage: run_headless.bat START END [NPROC] [OUT] [POOL]
exit /B 1
