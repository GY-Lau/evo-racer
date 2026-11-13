"""Track geometry and model validation, independent of CUDA runtime."""
import math
import numpy as np

WEIGHTS = 112
DT = 1 / 30
POINTS = 420
WIDTH, HEIGHT = 1100, 700


def geometry(kind):
    if kind not in ("flow", "technical"):
        raise ValueError("未知赛道")
    a = np.arange(POINTS, dtype=np.float32) / POINTS * (2 * math.pi)
    r = 1 + (.17 * np.sin(3 * a) + .065 * np.cos(5 * a) if kind == "technical" else .055 * np.sin(3 * a))
    points = np.stack((550 + 414 * r * np.cos(a), 350 + 231 * r * np.sin(a)), axis=1)
    road_width = 64 if kind == "technical" else 78
    rows, cols = math.ceil(HEIGHT / 3), math.ceil(WIDTH / 3)
    mask = np.zeros((rows, cols), dtype=bool)
    radius = road_width / 2 - 7
    for x, y in points:
        x0, x1 = max(0, int((x - radius) // 3)), min(cols, math.ceil((x + radius) / 3) + 1)
        y0, y1 = max(0, int((y - radius) // 3)), min(rows, math.ceil((y + radius) / 3) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx * 3 + 1.5 - x) ** 2 + (yy * 3 + 1.5 - y) ** 2 < radius ** 2
    return points, mask, road_width


def validate_model(data):
    if not isinstance(data, dict) or data.get("format") != "evo-racer-cuda-v1":
        raise ValueError("请选择 EVO CUDA 版导出的模型")
    weights, record = data.get("weights"), data.get("record")
    if data.get("track") not in ("flow", "technical") or not isinstance(weights, list) or len(weights) != WEIGHTS:
        raise ValueError("模型结构无效")
    if not all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 4 for v in weights):
        raise ValueError("模型权重无效")
    if type(record) not in (int, float) or not math.isfinite(record) or not 0 <= record <= 900:
        raise ValueError("模型成绩无效")
    return data
