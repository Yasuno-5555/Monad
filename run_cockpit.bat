@echo off
cd /d "%~dp0"
echo Starting Monad Cockpit...
python gui_main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ----------------------------------------------------------------
    echo Error: The application exited with code %ERRORLEVEL%.
    echo Please check the error message above.
    echo ----------------------------------------------------------------
    pause
)
