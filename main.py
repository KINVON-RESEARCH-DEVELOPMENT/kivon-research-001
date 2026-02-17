#!/usr/bin/env python3
"""Minimal GUI browser using tkinter + pywebview.

This provides a simple address bar and basic navigation controls.

Dependencies: `pywebview` (pip install pywebview). Optional: `psutil` to set CPU affinity.
"""

from __future__ import annotations

import os
import subprocess
import sys
import shutil
from typing import Tuple
import tkinter as tk
from tkinter import messagebox
from typing import Optional, Any

try:
    import webview  # type: ignore[reportMissingImports]
except Exception:
    webview = None

try:
    import psutil
except Exception:
    psutil = None


class SimpleBrowserApp:
    def __init__(self, root: tk.Tk, mem_limit_active: bool = False):
        self.root = root
        root.title("Kivon Simple Browser")

        toolbar = tk.Frame(root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.back_btn = tk.Button(toolbar, text="Back", command=self.go_back)
        self.back_btn.pack(side=tk.LEFT)

        self.forward_btn = tk.Button(toolbar, text="Forward", command=self.go_forward)
        self.forward_btn.pack(side=tk.LEFT)

        self.reload_btn = tk.Button(toolbar, text="Reload", command=self.reload_page)
        self.reload_btn.pack(side=tk.LEFT)

        self.address = tk.Entry(toolbar, width=60)
        self.address.pack(side=tk.LEFT, padx=4)

        go_btn = tk.Button(toolbar, text="Go", command=self.open_url)
        go_btn.pack(side=tk.LEFT)

        open_win_btn = tk.Button(toolbar, text="Open Window", command=self.open_window)
        open_win_btn.pack(side=tk.LEFT, padx=6)

        open_ext_btn = tk.Button(
            toolbar, text="Open in Chrome (persist)", command=self.open_in_chrome
        )
        open_ext_btn.pack(side=tk.LEFT, padx=6)

        # Memory-limit indicator (visible when application started with --mem-limit-active)
        self.mem_limit_active = mem_limit_active
        if self.mem_limit_active:
            try:
                self.mem_indicator = tk.Label(
                    toolbar, text="MEM LIMIT ACTIVE", fg="white", bg="red"
                )
                self.mem_indicator.pack(side=tk.RIGHT, padx=6)
            except Exception:
                self.mem_indicator = None

        self.status = tk.Label(root, text="Ready", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.webview_window: Optional[Any] = None
        self.webview_proc: Optional[subprocess.Popen] = None

    def set_status(self, text: str) -> None:
        self.status.config(text=text)

    def open_url(self) -> None:
        url = self.address.get().strip()
        if not url:
            return
        if not url.startswith(("http://", "https://", "about:")):
            url = "http://" + url
        if self.webview_window:
            try:
                # load_url is the recommended method
                self.webview_window.load_url(url)
                self.set_status(f"Loaded: {url}")
            except Exception:
                try:
                    self.webview_window.evaluate_js(f"location.href = '{url}';")
                except Exception:
                    self.set_status("Failed to load URL")
        else:
            # create new window
            self.address.delete(0, tk.END)
            self.address.insert(0, url)
            self.open_window()

    def go_back(self) -> None:
        if self.webview_window:
            try:
                self.webview_window.evaluate_js("history.back();")
            except Exception:
                self.set_status("Back failed")

    def go_forward(self) -> None:
        if self.webview_window:
            try:
                self.webview_window.evaluate_js("history.forward();")
            except Exception:
                self.set_status("Forward failed")

    def reload_page(self) -> None:
        if self.webview_window:
            try:
                self.webview_window.reload()
            except Exception:
                try:
                    self.webview_window.evaluate_js("location.reload();")
                except Exception:
                    self.set_status("Reload failed")

    def open_window(self) -> None:
        url = self.address.get().strip() or "https://example.com"
        if not webview:
            messagebox.showerror(
                "Missing dependency",
                "pywebview is required. Install with: pip install pywebview",
            )
            return

        # Launch webview in a separate process (pywebview must run on the process's main thread)
        runner = os.path.join(os.path.dirname(__file__), "webview_runner.py")
        if not os.path.exists(runner):
            messagebox.showerror(
                "Missing runner",
                "webview_runner.py not found; cannot open webview window.",
            )
            return

        try:
            proc = subprocess.Popen(
                [sys.executable, runner, url], cwd=os.path.dirname(__file__)
            )
            self.set_status(f"Opened webview (PID {proc.pid}) -> {url}")
            self.webview_proc = proc
            self.webview_window = None
        except Exception as e:
            self.set_status(f"Failed to open webview: {e}")

    def find_chrome_executable(self) -> Optional[str]:
        # Try PATH first
        for exe in (
            "google-chrome",
            "chrome",
            "chromium",
            "chromium-browser",
            "msedge",
        ):
            p = shutil.which(exe)
            if p:
                return p
        # Common Windows locations
        if sys.platform.startswith("win"):
            pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
            pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
            candidates = [
                os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
        return None

    def open_in_chrome(self) -> None:
        """Launch system Chrome/Edge with a persistent user-data-dir so logins persist."""
        url = self.address.get().strip() or "https://example.com"
        browser = self.find_chrome_executable()
        if not browser:
            messagebox.showerror(
                "Browser not found",
                "Could not find a Chrome/Chromium/MS Edge executable on PATH or common locations.",
            )
            return

        profile_dir = os.path.join(os.path.dirname(__file__), "profile")
        cache_dir = os.path.join(profile_dir, "Cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass

        flags = [
            f"--user-data-dir={profile_dir}",
            f"--disk-cache-dir={cache_dir}",
            "--ignore-gpu-blocklist",
            "--enable-gpu-rasterization",
            "--enable-quic",
        ]

        try:
            proc = subprocess.Popen(
                [browser] + flags + ["--new-window", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.set_status(
                f"Launched external browser (PID {proc.pid}) with persistent profile at {profile_dir}"
            )
            self.webview_proc = proc
        except Exception as e:
            messagebox.showerror("Launch failed", f"Failed to launch browser: {e}")

    def on_close(self) -> None:
        """Ask for confirmation and terminate the webview subprocess if running."""
        try:
            confirm = messagebox.askyesno("Quit", "Are you sure you want to quit?")
        except Exception:
            confirm = True
        if not confirm:
            return

        if self.webview_proc:
            try:
                if self.webview_proc.poll() is None:
                    self.set_status("Terminating webview process...")
                    self.webview_proc.terminate()
                    try:
                        self.webview_proc.wait(timeout=3)
                    except Exception:
                        try:
                            self.webview_proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> None:
    root = tk.Tk()
    mem_flag = "--mem-limit-active" in sys.argv
    app = SimpleBrowserApp(root, mem_limit_active=mem_flag)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.geometry("900x40")
    root.mainloop()


if __name__ == "__main__":
    main()
