@echo off
echo ========================================
echo   Log Analyzer Pro - GUI Launcher
echo ========================================
echo.
echo Starting Log Analyzer GUI...
echo.

REM Try C:\Python312 first (where packages are installed)
if exist "C:\Python312\python.exe" (
    echo Using Python: C:\Python312\python.exe
    C:\Python312\python.exe log_checker_gui.py
) else (
    REM Fallback to system python
    echo Using system Python
    python log_checker_gui.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: Failed to start GUI
    echo ========================================
    echo.
    echo Make sure you have installed all dependencies:
    echo   C:\Python312\python.exe -m pip install -r requirements.txt
    echo.
    echo And trained the model:
    echo   C:\Python312\python.exe train_model_gpu_ensemble.py
    echo.
    pause
)
