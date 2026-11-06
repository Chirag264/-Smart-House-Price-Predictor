@echo off
echo ========================================================
echo     Smart House Price Predictor - Streamlit Launcher
echo ========================================================
echo.

where streamlit >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Starting Streamlit App...
    streamlit run app.py
    goto end
)

if exist "C:\Users\pravi\AppData\Local\Python\pythoncore-3.14-64\python.exe" (
    echo Starting Streamlit via Python 3.14 environment...
    "C:\Users\pravi\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run app.py
    goto end
)

python -m streamlit run app.py

:end
pause

