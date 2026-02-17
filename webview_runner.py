#!/usr/bin/env python3
"""Run pywebview in its own process (must run on main thread).

Usage: python webview_runner.py <url>
"""
from __future__ import annotations
import sys
import signal

try:
    import webview
except Exception:
    print("pywebview is required. Install with: pip install pywebview")
    raise


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    win = webview.create_window("Kivon Browser", url, width=1024, height=700)

    # Ensure we exit cleanly on termination signals
    def _sigterm(signum, frame):
        try:
            sys.exit(0)
        except SystemExit:
            pass

    signal.signal(signal.SIGTERM, _sigterm)
    try:
        webview.start(gui="tkinter")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
