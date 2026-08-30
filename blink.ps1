$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$env:PYTHONPATH = $ScriptDir

$PythonExe = "C:\Users\polur\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

& $PythonExe "$ScriptDir\blink.py" @args
