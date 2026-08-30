@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHONPATH=%SCRIPT_DIR%"

if exist "C:\Users\polur\AppData\Local\Python\pythoncore-3.14-64\python.exe" (
    "C:\Users\polur\AppData\Local\Python\pythoncore-3.14-64\python.exe" "%SCRIPT_DIR%blink.py" %*
) else (
    python "%SCRIPT_DIR%blink.py" %*
)
