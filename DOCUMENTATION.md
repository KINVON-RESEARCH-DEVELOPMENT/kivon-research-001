# Kivon Research — Simple Browser & Launcher

Overview
--------

This small project contains two user-facing pieces:

- `cli boot.py`: a CLI helper to launch a Chromium-family browser with GPU- and network-friendly flags, and optional CPU affinity (requires `psutil`).
- `main.py`: a minimal GUI browser built with `tkinter` + `pywebview` for quick, local browsing within a simple window and toolbar.

Dependencies
------------

- Python 3.8+
- pywebview — for embedding a platform webview in the GUI: `pip install pywebview`
- psutil — optional, used by `cli boot.py` to set process CPU affinity: `pip install psutil`

Quick start
-----------

1. Install dependencies (recommended in a virtualenv):

```bash
pip install pywebview psutil
```

2. Run the simple GUI browser:

```bash
python main.py
```

Enter an address in the address bar and click `Open Window` or `Go` to navigate.

3. Use the CLI launcher to start your system Chromium/Edge with performance flags:

```bash
python "cli boot.py" --url https://example.com --cache-dir C:\\tmp\\chrome-cache --cache-size-mb 512
```

Arguments of note
-----------------

- `--browser-path`: Path to a Chromium-family browser executable. If omitted, `cli boot.py` will try to find a Chrome/Chromium/MS Edge executable on PATH or common install locations.
- `--no-gpu-flags`: Disable adding the GPU acceleration flags to the browser command-line.
- `--cpu-cores`: Comma-separated CPU cores or ranges to pin the browser process (requires `psutil`). Example: `--cpu-cores 0,1` or `--cpu-cores 0-3`.
- `--no-network-flags`: Disable the network flags added by default (QUIC, async DNS, larger disk cache settings).
- `--cache-dir` and `--cache-size-mb`: Configure disk cache directory and size for the launched browser.

Notes and limitations
---------------------

- The GUI browser (`main.py`) uses `pywebview` which wraps the platform's native web view (Edge WebView2 on recent Windows builds, WebKit on macOS, and GTK/WebKit on many Linux distributions). The look-and-feel is intentionally minimal.
- Network performance improvements offered by command-line flags are environment-dependent. QUIC requires server support; TCP Fast Open requires kernel and server support.
- CPU affinity changes require appropriate privileges and the `psutil` package. On Windows, changing affinity may require administrator privileges.
- This project is intentionally minimal — it is a small research/demo tool, not a full-featured browser.

Security
--------

- Only use these tools in trusted environments. Running a browser with experimental flags may expose instability or change security behavior.

Extending
---------

- To add a richer UI, add bookmarks, history, or tab support; consider moving to `pywebview` + a custom HTML-based UI or embedding Chromium via cefpython for full control.
- For automated testing, add a `requirements.txt` and small CI script to run a smoke test that starts `cli boot.py` and `main.py`.

Contact
-------

This code is part of a small research repository. If you want help extending it (e.g., adding bookmarks or a session manager), open an issue or request the changes.
