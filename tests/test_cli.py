"""
Unit tests for the BLINK CLI console (blink.py).
"""

import subprocess
import sys
from pathlib import Path
import pytest


def test_cli_help():
    res = subprocess.run([sys.executable, "blink.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "start" in res.stdout
    assert "stop" in res.stdout
    assert "host" in res.stdout
    assert "diagnose" in res.stdout


def test_cli_diagnose():
    res = subprocess.run([sys.executable, "blink.py", "diagnose"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "BLINK ENVIRONMENT & HARDWARE DIAGNOSTICS" in res.stdout
    assert "PyTorch Version" in res.stdout
    assert "[OK] FastAPI" in res.stdout


def test_cli_status():
    res = subprocess.run([sys.executable, "blink.py", "status"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "BLINK SYSTEM STATUS" in res.stdout
