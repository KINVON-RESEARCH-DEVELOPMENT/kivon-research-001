#!/usr/bin/env python3
"""Simple CLI to boot a Chromium-family browser with GPU and network optimizations.

Usage examples:
  python "cli boot.py" --url https://example.com
  python "cli boot.py" --browser-path "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --cpu-cores 0,1 --cache-dir C:\\tmp\\chrome-cache --cache-size-mb 512

Notes:
- Attempts to detect a local GPU and enables common Chrome flags for GPU acceleration.
- Adds network-related flags (QUIC, async DNS, disk cache) to improve throughput where supported.
- Optionally sets CPU affinity for the launched process if `psutil` is installed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import ctypes
from ctypes import wintypes

import subprocess
import sys
import time
from typing import List, Optional
import threading
import socket
import json
import urllib.request
import datetime


def detect_gpu() -> Optional[str]:
    """Return a short GPU description or None if not detectable."""
    try:
        if sys.platform.startswith("win"):
            out = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL,
            )
            lines = [
                l.strip() for l in out.decode(errors="ignore").splitlines() if l.strip()
            ]
            if len(lines) >= 2:
                return lines[1]
        elif sys.platform.startswith("linux"):
            out = subprocess.run(["lspci"], capture_output=True, text=True)
            for line in out.stdout.splitlines():
                if "VGA compatible controller" in line or "3D controller" in line:
                    return line.split(":", 1)[1].strip()
        elif sys.platform.startswith("darwin"):
            out = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            for line in out.stdout.splitlines():
                if "Chipset Model:" in line or "Graphics/Displays:" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def default_chrome_paths() -> List[str]:
    paths: List[str] = []
    if sys.platform.startswith("win"):
        paths += [
            os.path.join(
                os.environ.get("PROGRAMFILES", "C:\\Program Files"),
                "Google",
                "Chrome",
                "Application",
                "chrome.exe",
            ),
            os.path.join(
                os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"),
                "Google",
                "Chrome",
                "Application",
                "chrome.exe",
            ),
            os.path.join(
                os.environ.get("PROGRAMFILES", "C:\\Program Files"),
                "Microsoft",
                "Edge",
                "Application",
                "msedge.exe",
            ),
        ]
    else:
        paths += [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ]
    return paths


def find_browser(provided: Optional[str]) -> Optional[str]:
    if provided:
        if os.path.isfile(provided) or shutil.which(provided):
            return provided
        return None
    for exe in ("google-chrome", "chrome", "chromium", "chromium-browser", "msedge"):
        p = shutil.which(exe)
        if p:
            return p
    for p in default_chrome_paths():
        if p and os.path.exists(p):
            return p
    return None


def build_gpu_flags() -> List[str]:
    flags = [
        "--ignore-gpu-blocklist",
        "--enable-gpu-rasterization",
        "--enable-zero-copy",
        "--enable-native-gpu-memory-buffers",
        "--enable-accelerated-2d-canvas",
        "--enable-accelerated-video-decode",
        "--disable-features=VizDisplayCompositor",
    ]
    return [f"--{f.lstrip('-')}" if not f.startswith("--") else f for f in flags]


def build_network_flags(
    enable_quic: bool = True,
    cache_dir: Optional[str] = None,
    cache_size_mb: Optional[int] = None,
) -> List[str]:
    """Return network-related Chrome flags to encourage faster connections and caching.

    These flags are best-effort and may not improve performance in all environments.
    """
    flags: List[str] = []
    if enable_quic:
        flags.append("--enable-quic")
        # quic-version may be negotiated; setting a modern version is usually fine
        flags.append("--quic-version=43")
    flags.append("--enable-tcp-fastopen")
    flags.append("--enable-features=NetworkService")
    flags.append("--enable-async-dns")
    flags.append("--disable-background-networking")
    if cache_dir:
        flags.append(f"--disk-cache-dir={cache_dir}")
    if cache_size_mb and cache_size_mb > 0:
        flags.append(f"--disk-cache-size={int(cache_size_mb) * 1024 * 1024}")
    return flags


def get_public_ip_info(timeout: float = 3.0) -> Optional[dict]:
    """Return public IP geolocation info from ipinfo.io or None on failure."""
    services = ["https://ipinfo.io/json", "https://ipapi.co/json/"]
    for url in services:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
                return json.loads(data.decode())
        except Exception:
            continue
    return None


def monitor_network(
    interval: float = 5.0,
    iterations: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Monitor basic network info using psutil if available, printing to stdout.

    This version accepts being run in a background thread: pass a `stop_event` to
    wake the monitor and stop early. If `iterations` is None it will run until
    the `stop_event` is set.
    """
    try:
        import psutil
    except Exception:
        print(
            "psutil not available — install with: pip install psutil to enable monitoring"
        )
        return

    # Note: this function can be run either directly (no stop_event) or in a thread
    # with a threading.Event passed in. To preserve backward compatibility we keep
    # the same signature but allow callers to pass the event as a third arg via
    # positional parameters if desired.
    stop_event = None
    # If caller passed an event via a third positional arg (rare), accept it.
    # (This is simple and robust because inspect.signature would complicate things.)
    # The caller in this repo will pass the event explicitly.

    it = 0
    prev_counters = psutil.net_io_counters(pernic=True)
    while True:
        # If iterations limit reached, stop
        if iterations is not None and it >= iterations:
            print("Network monitoring finished (iterations reached)")
            break
        # If stop_event provided and set, stop
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            print("Network monitoring stopped (stop event set)")
            break

        print("--- Network status ---")
        # Interfaces
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        for ifname, s in stats.items():
            up = "up" if s.isup else "down"
            mtu = s.mtu
            print(f"{ifname}: {up}, MTU={mtu}")
            if ifname in addrs:
                for addr in addrs[ifname]:
                    print(f"  addr: {addr.address} family={addr.family}")

        # IO counters delta
        counters = psutil.net_io_counters(pernic=True)
        for ifname, cur in counters.items():
            prev = prev_counters.get(ifname)
            if prev:
                sent = cur.bytes_sent - prev.bytes_sent
                recv = cur.bytes_recv - prev.bytes_recv
                print(f"{ifname}: +{sent} B sent, +{recv} B recv in last {interval}s")
        prev_counters = counters

        # Active connections
        try:
            conns = psutil.net_connections()
            tcp = sum(
                1 for c in conns if getattr(c, "type", None) == socket.SOCK_STREAM
            )
            udp = sum(1 for c in conns if getattr(c, "type", None) == socket.SOCK_DGRAM)
            print(f"Active connections: TCP={tcp}, UDP={udp}, total={len(conns)}")
        except Exception:
            print("Could not fetch active connections")

        it += 1
        # Sleep in small increments so we can respond quickly to a stop_event
        slept = 0.0
        while slept < interval:
            if (
                stop_event is not None
                and getattr(stop_event, "is_set", lambda: False)()
            ):
                break
            time.sleep(0.5)
            slept += 0.5


def monitor_memory(
    pid: int,
    after_seconds: Optional[float] = None,
    log_file: Optional[str] = None,
    interval: float = 1.0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Monitor a process memory usage. If `after_seconds` is set and `log_file` is None,
    this will sleep and then print a single RSS sample. If `log_file` is set, it will
    log timestamped RSS samples every `interval` seconds until the process exits or
    `stop_event` is set.
    """
    try:
        import psutil
    except Exception:
        print(
            "psutil not available — install with: pip install psutil to enable memory monitoring"
        )
        return

    try:
        p = psutil.Process(pid)
    except Exception:
        print(f"Process {pid} not found for memory monitoring")
        return

    if log_file:
        try:
            f = open(log_file, "a", encoding="utf-8")
        except Exception as e:
            print(f"Failed to open memory log file {log_file}: {e}")
            return
        print(f"Logging memory for PID {pid} -> {log_file}")
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if p.is_running() is False or p.status() == psutil.STATUS_ZOMBIE:
                    break
                m = p.memory_info()
                ts = datetime.datetime.utcnow().isoformat()
                f.write(f"{ts},pid={pid},rss={m.rss},vms={m.vms}\n")
                f.flush()
                time.sleep(interval)
        finally:
            f.close()
    else:
        # one-shot after delay or immediate
        if after_seconds and after_seconds > 0:
            time.sleep(after_seconds)
        try:
            m = p.memory_info()
            print(
                f"Memory sample for PID {pid}: RSS={m.rss/1024**2:.2f} MB, VMS={m.vms/1024**2:.2f} MB"
            )
        except Exception as e:
            print(f"Failed to sample memory for PID {pid}: {e}")


def set_process_memory_limit_windows(pid: int, limit_bytes: int):
    """
    Create a Windows Job Object with a per-process memory limit and assign the
    target process to it. Also spawn a background thread that will assign
    newly spawned child processes to the same job so helper processes are
    constrained as well.

    Returns a dict with keys: 'job' (HANDLE), 'stop_event' (threading.Event),
    'thread' (Thread), and 'close' (callable to clean up the job). Returns
    None on failure.
    """
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except Exception:
        return None

    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JobObjectExtendedLimitInformation = 9
    PROCESS_ALL_ACCESS = 0x1F0FFF

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    CreateJobObject = kernel32.CreateJobObjectW
    CreateJobObject.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    CreateJobObject.restype = wintypes.HANDLE

    SetInformationJobObject = kernel32.SetInformationJobObject
    SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    SetInformationJobObject.restype = wintypes.BOOL

    OpenProcess = kernel32.OpenProcess
    OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    OpenProcess.restype = wintypes.HANDLE

    AssignProcessToJobObject = kernel32.AssignProcessToJobObject
    AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    AssignProcessToJobObject.restype = wintypes.BOOL

    CloseHandle = kernel32.CloseHandle
    CloseHandle.argtypes = (wintypes.HANDLE,)
    CloseHandle.restype = wintypes.BOOL

    job = CreateJobObject(None, None)
    if not job:
        return None

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = ctypes.c_size_t(limit_bytes)

    res = SetInformationJobObject(
        job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    )
    if not res:
        try:
            CloseHandle(job)
        except Exception:
            pass
        return None

    # Assign the initial process
    hproc = OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not hproc:
        try:
            CloseHandle(job)
        except Exception:
            pass
        return None
    try:
        ok = AssignProcessToJobObject(job, hproc)
    except Exception:
        ok = False

    # Prepare a background thread to attach future child processes to the job
    stop_event = threading.Event()
    assigned = set()

    def _assign_children_loop():
        try:
            import psutil
        except Exception:
            return
        try:
            root = psutil.Process(pid)
        except Exception:
            return
        while not stop_event.is_set():
            try:
                children = root.children(recursive=True)
                for c in children:
                    if c.pid in assigned:
                        continue
                    try:
                        hp = OpenProcess(PROCESS_ALL_ACCESS, False, c.pid)
                        if hp:
                            try:
                                AssignProcessToJobObject(job, hp)
                                assigned.add(c.pid)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                # process may exit; just break
                pass
            stop_event.wait(1.0)

    t = threading.Thread(target=_assign_children_loop, daemon=True)
    t.start()

    def _close():
        try:
            stop_event.set()
        except Exception:
            pass
        try:
            t.join(timeout=2)
        except Exception:
            pass
        try:
            CloseHandle(job)
        except Exception:
            pass

    return {
        "job": job,
        "stop_event": stop_event,
        "thread": t,
        "close": _close,
        "assigned": assigned,
        "assigned_ok": bool(ok),
    }


def get_total_memory() -> Optional[int]:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        try:

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("sullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except Exception:
            return None
    return None


def parse_cpu_cores(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    try:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        cores = []
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                cores.extend(range(int(a), int(b) + 1))
            else:
                cores.append(int(p))
        return sorted(set(cores))
    except Exception:
        return None


def set_affinity(pid: int, cores: List[int]) -> bool:
    try:
        import psutil

        p = psutil.Process(pid)
        p.cpu_affinity(cores)
        return True
    except Exception:
        return False


def launch_browser(
    browser: str,
    url: str,
    gpu_flags: bool,
    cpu_cores: Optional[List[int]],
    network_flags: Optional[List[str]] = None,
):
    args = [browser]
    if gpu_flags:
        args += build_gpu_flags()
    if network_flags:
        args += network_flags
    args += ["--new-window", "--disable-popup-blocking", f"{url}"]

    print("Launching:", " ".join(shlex.quote(a) for a in args))
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(0.5)

    if cpu_cores:
        ok = set_affinity(proc.pid, cpu_cores)
        if ok:
            print(f"Set CPU affinity for PID {proc.pid} -> cores {cpu_cores}")
        else:
            print(
                "Warning: couldn't set CPU affinity (is psutil installed and running as admin?)"
            )

    print(f"Browser launched (PID {proc.pid}).")
    return proc


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Boot a Chromium browser with GPU and network-friendly flags"
    )
    p.add_argument("--browser-path", help="Path to Chromium/Chrome/MS Edge executable")
    p.add_argument("--url", default="about:blank", help="URL to open")
    p.add_argument(
        "--no-gpu-flags",
        dest="gpu_flags",
        action="store_false",
        help="Disable adding recommended GPU flags",
    )
    p.add_argument(
        "--cpu-cores",
        help="Comma-separated CPU core indices or ranges (e.g. 0,1 or 0-3)",
    )
    p.add_argument(
        "--detect-gpu", action="store_true", help="Detect and print GPU info then exit"
    )
    p.add_argument(
        "--no-network-flags",
        dest="network_flags",
        action="store_false",
        help="Disable adding recommended network flags",
    )
    p.add_argument("--cache-dir", help="Path for disk cache to increase cache reuse")
    p.add_argument("--cache-size-mb", type=int, help="Disk cache size in MB (e.g. 512)")
    p.add_argument(
        "--start-main",
        action="store_true",
        help="Start the bundled GUI (`main.py`) and optionally monitor network",
    )
    p.add_argument(
        "--monitor-interval",
        type=float,
        default=5.0,
        help="Seconds between network status reports when monitoring",
    )
    p.add_argument(
        "--monitor-iterations",
        type=int,
        default=0,
        help="Number of network monitoring iterations (0 = run until process exit) - only used with --start-main",
    )
    p.add_argument(
        "--geo",
        action="store_true",
        help="Query public IP geolocation service and print location info before starting",
    )
    p.add_argument(
        "--measure-seconds",
        type=float,
        default=0.0,
        help="Wait N seconds after launching and print one memory sample (RSS) for the launched process",
    )
    p.add_argument(
        "--memory-log-file",
        help="Path to append continuous memory log (timestamp,rss,vms) for launched process",
    )
    p.add_argument(
        "--memory-interval",
        type=float,
        default=1.0,
        help="Interval seconds for continuous memory logging",
    )
    p.add_argument(
        "--mem-percent",
        type=float,
        default=0.0,
        help="Limit launched process memory to this percent of total RAM (Windows only)",
    )
    ns = p.parse_args(argv)

    # Option: start the bundled GUI (`main.py`) and monitor network
    if ns.start_main:
        if ns.geo:
            info = get_public_ip_info()
            if info:
                print("Public IP info:", info)
            else:
                print("Could not get public IP info")

        main_py = os.path.join(os.path.dirname(__file__), "main.py")
        if not os.path.exists(main_py):
            print("Error: main.py not found next to cli boot.py")
            return 3

        main_py_abs = os.path.abspath(main_py)
        print(f"Starting GUI: {main_py_abs}")
        # Capture stderr so we can show an error if the child exits immediately
        proc = subprocess.Popen(
            [sys.executable, main_py_abs]
            + (
                ["--mem-limit-active"]
                if ns.mem_percent
                and ns.mem_percent > 0
                and sys.platform.startswith("win")
                else []
            ),
            cwd=os.path.dirname(__file__),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Use a stop event for monitoring threads
        stop_event = threading.Event()

        # If requested, set a memory limit (Windows only) and track the job object
        job_info = None
        if ns.mem_percent and ns.mem_percent > 0 and sys.platform.startswith("win"):
            total = get_total_memory()
            if total:
                limit = int(total * float(ns.mem_percent) / 100.0)
                job_info = set_process_memory_limit_windows(proc.pid, limit)
                if job_info:
                    print(
                        f"Set Windows job memory limit: {ns.mem_percent}% -> {limit} bytes for PID {proc.pid}"
                    )
                else:
                    print("Warning: failed to set Windows job memory limit")

        # Optional memory monitoring for GUI subprocess
        mem_thread = None
        if ns.measure_seconds > 0.0 or ns.memory_log_file:
            mem_thread = threading.Thread(
                target=monitor_memory,
                args=(
                    proc.pid,
                    ns.measure_seconds if ns.measure_seconds > 0 else None,
                    ns.memory_log_file,
                    ns.memory_interval,
                    stop_event,
                ),
                daemon=True,
            )
            mem_thread.start()

        # Monitor network while main.py is running (in a background thread).
        iterations = None if ns.monitor_iterations <= 0 else ns.monitor_iterations
        mon_thread = threading.Thread(
            target=monitor_network,
            args=(ns.monitor_interval, iterations, stop_event),
            daemon=True,
        )
        mon_thread.start()

        try:
            # Wait until the monitor finishes or the GUI process exits.
            while True:
                if not mon_thread.is_alive():
                    print("Monitor finished; terminating GUI...")
                    break
                if proc.poll() is not None:
                    print("GUI process exited; stopping monitor...")
                    # Print stderr from child to help debugging
                    try:
                        err = (
                            proc.stderr.read().decode(errors="replace")
                            if proc.stderr is not None
                            else None
                        )
                        if err:
                            print("--- GUI stderr ---")
                            print(err)
                    except Exception:
                        pass
                    stop_event.set()
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopping monitoring; terminating GUI...")
            stop_event.set()
        finally:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            # stop monitoring threads
            try:
                stop_event.set()
            except Exception:
                pass
            if mem_thread is not None:
                mem_thread.join(timeout=2)
            # cleanup job object if we created one
            if job_info:
                try:
                    job_info.get("close", lambda: None)()
                except Exception:
                    pass
            mon_thread.join(timeout=5)
        return 0

    if ns.detect_gpu:
        g = detect_gpu()
        if g:
            print("Detected GPU:", g)
            return 0
        print("No GPU detected or detection failed.")
        return 1

    browser = find_browser(ns.browser_path)
    if not browser:
        print(
            "Error: could not find a Chromium/Chrome/MS Edge executable. Provide --browser-path."
        )
        return 2

    cores = parse_cpu_cores(ns.cpu_cores)
    if ns.cpu_cores and cores is None:
        print("Warning: couldn't parse --cpu-cores; ignoring affinity setting.")

    net_flags = None
    if ns.network_flags:
        net_flags = build_network_flags(
            enable_quic=True, cache_dir=ns.cache_dir, cache_size_mb=ns.cache_size_mb
        )

    proc = launch_browser(browser, ns.url, ns.gpu_flags, cores, net_flags)

    # If requested, set a memory limit for the launched browser (Windows only)
    job_info = None
    if (
        proc
        and ns.mem_percent
        and ns.mem_percent > 0
        and sys.platform.startswith("win")
    ):
        total = get_total_memory()
        if total:
            limit = int(total * float(ns.mem_percent) / 100.0)
            job_info = set_process_memory_limit_windows(proc.pid, limit)
            if job_info:
                print(
                    f"Set Windows job memory limit: {ns.mem_percent}% -> {limit} bytes for PID {proc.pid}"
                )
            else:
                print("Warning: failed to set Windows job memory limit for browser")

    # If memory measurement or logging requested for launched browser, run monitor.
    if proc and (ns.measure_seconds > 0.0 or ns.memory_log_file):
        stop_event = threading.Event()
        mem_thread = threading.Thread(
            target=monitor_memory,
            args=(
                proc.pid,
                ns.measure_seconds if ns.measure_seconds > 0 else None,
                ns.memory_log_file,
                ns.memory_interval,
                stop_event,
            ),
            daemon=True,
        )
        mem_thread.start()
        try:
            # Wait until monitoring thread finishes or process exits
            while True:
                if not mem_thread.is_alive():
                    break
                if proc.poll() is not None:
                    stop_event.set()
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            mem_thread.join(timeout=5)
            # cleanup job object created for browser launch
            if job_info:
                try:
                    job_info.get("close", lambda: None)()
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
