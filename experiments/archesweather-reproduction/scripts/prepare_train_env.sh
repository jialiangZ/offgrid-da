#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/geoarches"
cd "$REPO"
/home/jlzhang/.local/bin/uv sync --frozen
/home/jlzhang/.local/bin/uv pip install --python "$REPO/.venv/bin/python" --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
"$REPO/.venv/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
"$REPO/.venv/bin/python" -c "import swanlab; from swanlab.integration.pytorch_lightning import SwanLabLogger; print('SwanLab', swanlab.__version__, SwanLabLogger.__name__)"
