#!/usr/bin/env python3
"""
BLINK Command-Line Management Console.
Provides unified operational commands:
  blink start / launch    - Starts the BLINK server (default: localhost:8000)
  blink stop              - Terminates the running BLINK server
  blink host              - Hosts the dashboard on local Wi-Fi / LAN (0.0.0.0) with broadcast URL
  blink status            - Checks server health, PID, and live status
  blink diagnose          - Runs comprehensive system, hardware, and dependency diagnostics
"""

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import urllib.request


WORKSPACE_ROOT = Path(__file__).resolve().parent
PID_FILE = WORKSPACE_ROOT / ".blink.pid"
LOG_FILE = WORKSPACE_ROOT / "data" / "blink_server.log"
DEFAULT_PORT = 8000


def get_local_ip() -> str:
    """Detects the primary local Wi-Fi / LAN IPv4 address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to public DNS without sending packets to find local routing IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a TCP port is currently listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def read_pid() -> Optional[int]:
    """Reads PID from PID file if it exists and is alive."""
    if not PID_FILE.exists():
        return None
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        # Check if process is still running
        if platform.system() == "Windows":
            res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
            if str(pid) in res.stdout:
                return pid
        else:
            os.kill(pid, 0)
            return pid
    except Exception:
        pass
    return None


def write_pid(pid: int) -> None:
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def remove_pid() -> None:
    if PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except Exception:
            pass


def kill_process_on_port(port: int) -> bool:
    """Terminates any process currently listening on the specified port."""
    if not is_port_in_use(port):
        return True
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True, text=True)
            for line in out.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in parts:
                    pid = parts[-1]
                    subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
            time.sleep(0.8)
        except Exception:
            pass
    else:
        try:
            subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
            time.sleep(0.5)
        except Exception:
            pass
    return not is_port_in_use(port)


# ------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------
def cmd_start(args: argparse.Namespace) -> None:
    """Starts the BLINK FastAPI / Uvicorn server."""
    port = args.port or DEFAULT_PORT
    host = args.host or "127.0.0.1"
    daemon = args.daemon

    existing_pid = read_pid()
    if existing_pid:
        print(f"[BLINK] Server process {existing_pid} active. Re-initializing...")
        cmd_stop(args)

    if is_port_in_use(port, host):
        print(f"[BLINK] Port {port} is in use. Reclaiming port...")
        kill_process_on_port(port)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    print("=" * 65)
    print("  PROJECT BLINK - SATELLITE TEMPORAL NOWCASTING CONSOLE")
    print("=" * 65)
    print(f"  Binding Host   : {host}")
    print(f"  Port           : {port}")
    print(f"  Local Access   : http://localhost:{port}")
    if host == "0.0.0.0":
        local_ip = get_local_ip()
        print(f"  Wi-Fi Access   : http://{local_ip}:{port}")
    print("=" * 65)

    if daemon:
        print(f"[BLINK] Launching background daemon (logs -> {LOG_FILE})...")
        with open(LOG_FILE, "a") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(WORKSPACE_ROOT),
                env=env,
                stdout=log_f,
                stderr=log_f,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0,
            )
        write_pid(proc.pid)
        time.sleep(1.5)
        if proc.poll() is None:
            print(f"[BLINK] Server successfully started in background [PID: {proc.pid}]")
        else:
            print("[BLINK] Failed to start server in background. Check logs at data/blink_server.log")
    else:
        print("[BLINK] Starting server in foreground (Ctrl+C to stop)...\n")
        try:
            proc = subprocess.Popen(cmd, cwd=str(WORKSPACE_ROOT), env=env)
            write_pid(proc.pid)
            proc.wait()
        except KeyboardInterrupt:
            print("\n[BLINK] Shutting down...")
        finally:
            remove_pid()


def cmd_host(args: argparse.Namespace) -> None:
    """Hosts the BLINK console on local Wi-Fi / LAN (0.0.0.0)."""
    args.host = "0.0.0.0"
    args.port = args.port or DEFAULT_PORT
    cmd_start(args)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stops the running BLINK server."""
    pid = read_pid()
    if not pid:
        # Fallback check port
        port = args.port or DEFAULT_PORT
        if is_port_in_use(port):
            print(f"[BLINK] Found active process on port {port}. Attempting cleanup...")
            if platform.system() == "Windows":
                subprocess.run(f"for /f \"tokens=5\" %a in ('netstat -aon ^| find \":{port}\"') do taskkill /f /pid %a", shell=True, capture_output=True)
            print("[BLINK] Stopped active listeners on port.")
        else:
            print("[BLINK] No active BLINK server found.")
        remove_pid()
        return

    print(f"[BLINK] Terminating BLINK server process [PID: {pid}]...")
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, 15)
        remove_pid()
        print("[BLINK] Server stopped successfully.")
    except Exception as e:
        print(f"[BLINK] Could not terminate process {pid}: {e}")


def cmd_status(args: argparse.Namespace) -> None:
    """Checks live server status and API health."""
    port = args.port or DEFAULT_PORT
    pid = read_pid()
    print("=" * 55)
    print("  BLINK SYSTEM STATUS")
    print("=" * 55)
    print(f"  PID File       : {pid if pid else 'None (Inactive)'}")
    print(f"  Port {port} Status: {'Listening' if is_port_in_use(port) else 'Closed'}")

    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/health", headers={"User-Agent": "BLINK-CLI/1.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  Service Status : Operational ({data.get('status')})")
            print(f"  Compute Device : {data.get('device')}")
            print(f"  CUDA Available : {data.get('cuda_available')}")
            print(f"  PyTorch        : {data.get('torch_version')}")
    except Exception:
        print("  API Gateway    : Offline / Unreachable")
    print("=" * 55)


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Runs comprehensive diagnostics across dependencies, hardware, and configuration."""
    print("=" * 65)
    print("  BLINK ENVIRONMENT & HARDWARE DIAGNOSTICS")
    print("=" * 65)
    print(f"  Operating System : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python Version   : {sys.version.split()[0]} ({sys.executable})")

    # PyTorch & Hardware
    try:
        import torch
        print(f"  PyTorch Version  : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"  CUDA Available   : {cuda_avail}")
        if cuda_avail:
            print(f"  GPU Device       : {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  Total VRAM       : {vram_gb:.2f} GB")
        else:
            print("  Inference Mode   : CPU Optimized (Graceful Fallback)")
    except ImportError:
        print("  [ERROR] PyTorch is NOT installed in this environment.")

    # Dependencies Check
    print("\n  Dependency Status:")
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("scipy", "SciPy (NETRA Morphological Nowcasting)"),
        ("PIL", "Pillow (Image IO)"),
        ("requests", "Requests (MOSDAC HTTP API)"),
        ("httpx", "HTTPX (Async Test Client)"),
        ("s3fs", "S3FS (NOAA/AWS S3 Client)"),
        ("h5py", "h5py (MOSDAC HDF5 L1B/L2 Parser)"),
    ]

    for mod, label in packages:
        try:
            __import__(mod)
            print(f"    [OK] {label}")
        except ImportError:
            print(f"    [MISSING] {label}")

    # MOSDAC Config Check
    config_file = WORKSPACE_ROOT / "config.json"
    print("\n  MOSDAC Credentials:")
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
            user = cfg.get("user_credentials", {}).get("username/email") or cfg.get("user_credentials", {}).get("username")
            if user and user != "your_username":
                print(f"    [Configured] Account: {user}")
            else:
                print("    [Unconfigured] Default placeholder detected in config.json")
        except Exception:
            print("    [Warning] Could not parse config.json")
    else:
        print("    [Missing] config.json not found in root")

    # Network Ports
    print("\n  Network Diagnostics:")
    print(f"    Local Host IP   : {get_local_ip()}")
    print(f"    Port 8000 Free  : {not is_port_in_use(8000)}")
    print(f"    Port 8080 Free  : {not is_port_in_use(8080)}")
    print("=" * 65)


# ------------------------------------------------------------------------------
# Entrypoint & CLI Parser
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="blink",
        description="Project BLINK CLI Management Console",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start / launch
    p_start = subparsers.add_parser("start", aliases=["launch"], help="Start BLINK web server")
    p_start.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8000)")
    p_start.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_start.add_argument("-d", "--daemon", action="store_true", help="Run server in background")

    # host
    p_host = subparsers.add_parser("host", help="Host dashboard on local Wi-Fi network (0.0.0.0)")
    p_host.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8000)")
    p_host.add_argument("-d", "--daemon", action="store_true", help="Run server in background")

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop running BLINK server")
    p_stop.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to clear")

    # status
    p_stat = subparsers.add_parser("status", help="Check server health and PID")
    p_stat.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to check")

    # diagnose
    subparsers.add_parser("diagnose", help="Run hardware and environment diagnostics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cmd_map = {
        "start": cmd_start,
        "launch": cmd_start,
        "host": cmd_host,
        "stop": cmd_stop,
        "status": cmd_status,
        "diagnose": cmd_diagnose,
    }

    handler = cmd_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
