@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHONUTF8=1

echo [1/2] Installing dependencies...
py -3 -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    echo Try running:
    echo py -3 -m pip install Flask requests pypdf pdfplumber pandas openpyxl chardet
    echo.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting...
py -3 app.py

echo.
echo The program has stopped. Error code: %errorlevel%
pause
