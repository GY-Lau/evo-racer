#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
EVO_PYTHON="${EVO_PYTHON:-python3}"
if [[ "$EVO_PYTHON" == python3 && -x .venv/bin/python ]]; then
    EVO_PYTHON="$PWD/.venv/bin/python"
fi
echo '--- System ---'
uname -m
head -n 1 /etc/nv_tegra_release 2>/dev/null || true
dpkg-query -W nvidia-jetpack nvidia-l4t-core python3 2>/dev/null || true
if command -v nvcc >/dev/null; then
    nvcc --version
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    /usr/local/cuda/bin/nvcc --version
fi
echo '--- Python / PyTorch ---'
"$EVO_PYTHON" - <<'PY'
import sys
print('Python:', sys.version)
print('Executable:', sys.executable)
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('Built CUDA:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit('CUDA 不可用，停止部署。')
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.ones((64,64), device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    value = y[0,0].item()
    if y.device.type != 'cuda' or value != 64:
        raise SystemExit('CUDA 矩阵乘法校验失败，停止部署。')
    print('CUDA matmul:', value, 'device:', y.device)
except ImportError as exc:
    raise SystemExit(f'PyTorch 无法导入，停止部署：{exc}') from exc
PY
