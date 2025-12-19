@echo off
echo Starting Monad Studio Visual Builder...
set PYTHONPATH=%~dp0
python -m monad.gui.app
pause
