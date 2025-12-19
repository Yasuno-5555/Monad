@echo off
echo Starting Monad Studio Visual Builder...
:: Point to parent directory for PYTHONPATH
set PYTHONPATH=%~dp0..
python -m monad.gui.app
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Application exited with code %ERRORLEVEL%
    pause
)
