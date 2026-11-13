#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
EVO_TARGET="${1:-hai@192.168.110.186}"
if [[ "$EVO_TARGET" == -* || "$EVO_TARGET" == *[!a-zA-Z0-9@._-]* ]]; then
  echo '无效 SSH 目标' >&2; exit 2
fi
echo "复制项目到 ${EVO_TARGET}:~/evo-racer（由 SSH 提示输入密码）"
ssh "$EVO_TARGET" 'mkdir -p ~/evo-racer/dist ~/evo-racer/tests ~/evo-racer/data ~/evo-racer/tools'
scp f1_config.py grand_prix.py server.py cuda_training.py race_common.py run.sh diagnose-jetson.sh README.md requirements-jetson.txt evo-racer.service "$EVO_TARGET:evo-racer/"
scp dist/index.html dist/style.css dist/display.js dist/motion.js dist/camera.js dist/app.js "$EVO_TARGET:evo-racer/dist/"
scp tools/benchmark_arena.py "$EVO_TARGET:evo-racer/tools/"
scp tests/test_app.cjs tests/test_performance_physics.py tests/test_holdout.cjs tests/test_camera.cjs tests/test_motion.cjs tests/test_grand_prix.py tests/test_common.py tests/test_cuda.py "$EVO_TARGET:evo-racer/tests/"
scp -r data/circuits data/LICENSE-f1-circuits.md "$EVO_TARGET:evo-racer/data/"
ssh "$EVO_TARGET" 'set -e
cd ~/evo-racer
bash diagnose-jetson.sh
if [ -x .venv/bin/python ]; then .venv/bin/python -m unittest discover -s tests -v; else python3 -m unittest discover -s tests -v; fi
mkdir -p ~/.config/systemd/user
cp evo-racer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable evo-racer.service
systemctl --user restart evo-racer.service
systemctl --user is-active evo-racer.service'
