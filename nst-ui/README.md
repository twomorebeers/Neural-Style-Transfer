# NST UI (Next.js)

Local web interface for our existing Python neural style transfer backend.

## What this app does

- Upload content and style images
- Select one or more backbones (`vgg16`, `vgg19`, `resnet50`, `inception_v3`, `squeezenet1_1`)
- Configure weights / steps / device / init mode
- Calls `nst_project_code/nst_compare.py`
- Shows generated images and downloadable CSV histories

## Requirements

- Node.js 20+
- A working Python environment with dependencies from:
  - `nst_project_code/requirements.txt`

## Cross-platform compatibility

- The UI backend auto-detects Python in this order:
   1. `PYTHON_BIN` env var (if set)
   2. `../.venv/bin/python` (macOS/Linux)
   3. `../.venv/Scripts/python.exe` (Windows)
   4. `python3` (macOS/Linux) or `python` (Windows)
- Works on Apple Silicon (ARM64), Intel macOS, Linux x64/ARM, and Windows x64, as long as the selected Python has `torch` and `torchvision` installed.
- For best reproducibility across teammates, each machine should create its own local `.venv` and install Python packages there.

## Setup

From this directory (`nst-ui`):

1. Install dependencies:
   - `npm install`
2. (Optional) set Python command if needed:
   - macOS/Linux default: `python3`
   - override via env: `PYTHON_BIN=/absolute/or/resolved/python`
   - Windows PowerShell example: `$env:PYTHON_BIN="C:\path\to\python.exe"`
3. Start dev server:
   - `npm run dev`

Then open <http://localhost:3000>

## Notes

- The backend stores temporary runtime files in `nst-ui/.runtime/`.
- All image/CSV artifacts are served via API endpoints.
- This frontend logic is designed for local execution.
