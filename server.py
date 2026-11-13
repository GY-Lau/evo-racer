#!/usr/bin/env python3
"""Jetson CUDA trainer and dependency-free HTTP display server."""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def cuda_environment():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("未安装 PyTorch。请先在 Jetson 按 JetPack 版本安装 NVIDIA CUDA 版 PyTorch；参阅 README.md。") from exc
    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA 不可用（torch={torch.__version__}, built_cuda={torch.version.cuda}）。程序已停止，不会回退 CPU。请在 Jetson 上运行并安装匹配 JetPack 的 PyTorch。")
    # Execute real GPU work before claiming CUDA is active.
    with torch.no_grad():
        x = torch.ones((64, 64), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        if y.device.type != "cuda" or y[0, 0].item() != 64:
            raise SystemExit("CUDA 计算验证失败")
    return {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "torch": torch.__version__, "device": "cuda:0", "verified": True}


class Trainer:
    def __init__(self, args, device_info):
        self.args = args
        self.device_info = device_info
        self.commands = queue.Queue(maxsize=32)
        self.lock = threading.Lock()
        self.state = {"ready": False}
        self.model = None
        self.stop = threading.Event()
        self.error = None

    def publish(self, sim, *, paused, speed, steps_per_second, message="", trial=None):
        import torch
        state = (trial if trial is not None else sim).snapshot()
        model = sim.export_model()
        trial_status = None
        if trial is not None:
            stopped = state['alive'] == 0
            trial_status = {"stopped": stopped, "reason": trial.end_reason(), "laps": max(0., float(trial.progress[0].item())) / 420}
            state.update(generation=sim.info.generation, record=sim.info.record, lastBest=sim.info.last_best, history=list(sim.history))
        state.update(ready=True, paused=paused, speed=speed, mutation=sim.mutation, device=self.device_info, stepsPerSecond=round(steps_per_second, 1), gpuMemoryMB=round(torch.cuda.memory_allocated() / 1024 ** 2, 1), message=message, mode="champion" if trial is not None else "training", trial=trial_status)
        with self.lock:
            self.state, self.model = state, model

    def run(self):
        try:
            self._run()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = str(exc)
            with self.lock:
                self.state = {"ready": False, "error": self.error}

    def _run(self):
        from cuda_training import CudaEvolution
        sim = CudaEvolution(population=self.args.population, seed=self.args.seed)
        save_path = Path(self.args.checkpoint)
        message = "Jetson CUDA 已连接，正在训练。"
        if save_path.exists():
            sim.import_model(json.loads(save_path.read_text()))
            message = "已从 Jetson 存档恢复最佳大脑。"
        paused, speed = False, 1
        trial, training_paused = None, False
        measured_steps, rate = 0, 0.
        clock_start = last_publish = time.monotonic()
        self.publish(sim, paused=paused, speed=speed, steps_per_second=rate, message=message)
        while not self.stop.is_set():
            tick_start = time.monotonic()
            while True:
                try:
                    command, done, result = self.commands.get_nowait()
                except queue.Empty:
                    break
                try:
                    action = command.get("action")
                    if trial is not None and action in ("mutation", "next", "reset", "import"):
                        raise ValueError("请先返回进化训练，再修改训练实验")
                    if action == "pause": paused = bool(command["value"])
                    elif action == "speed":
                        if command.get("value") not in (1, 3, 10, 30): raise ValueError("无效倍率")
                        speed = command["value"]
                    elif action in ("mode", "restart_trial"):
                        mode = command.get("value") if action == "mode" else "champion"
                        if mode not in ("training", "champion"):
                            raise ValueError("无效运行模式")
                        if action == "restart_trial" and trial is None:
                            raise ValueError("请先进入冠军试跑")
                        if mode == "champion" and (trial is None or action == "restart_trial"):
                            model = sim.export_model()
                            if model is None:
                                raise ValueError("还没有最佳大脑，请先训练片刻")
                            candidate = CudaEvolution(population=1, device=sim.device, trial=True)
                            candidate.import_model(model)
                            if trial is None:
                                training_paused = paused
                                self.save_checkpoint(sim, save_path)
                            trial, paused = candidate, False
                        elif mode == "training" and trial is not None:
                            trial, paused = None, training_paused
                        measured_steps, rate, clock_start = 0, 0., time.monotonic()
                    elif action == "mutation":
                        value = float(command["value"])
                        if not .02 <= value <= .5: raise ValueError("无效变异概率")
                        sim.mutation = value
                    elif action == "next": sim.next_generation()
                    elif action == "reset":
                        sim = CudaEvolution(kind=command.get("kind", sim.kind), population=self.args.population, mutation=sim.mutation, seed=time.time_ns() % (2 ** 32))
                        save_path.unlink(missing_ok=True)
                    elif action == "import": sim.import_model(command["model"])
                    else: raise ValueError("未知命令")
                    if action in ("import", "next"):
                        self.save_checkpoint(sim, save_path)
                    self.publish(sim, paused=paused, speed=speed, steps_per_second=rate, trial=trial)
                    result["ok"] = True
                except (ValueError, KeyError, TypeError) as exc:
                    result["error"] = str(exc)
                finally:
                    done.set()
            # 1x targets 30 physics steps/s. Higher rates are capped by hardware.
            if not paused:
                for _ in range(speed):
                    active = trial if trial is not None else sim
                    if trial is not None and trial.finished():
                        break
                    active.step()
                    measured_steps += 1
                    if trial is None and sim.ticks % 4 == 0 and sim.finished():
                        sim.next_generation(reason=sim.end_reason())
                        self.save_checkpoint(sim, save_path)
            now = time.monotonic()
            if now - clock_start >= 1:
                rate = measured_steps / (now - clock_start)
                measured_steps, clock_start = 0, now
            if now - last_publish >= .1:
                self.publish(sim, paused=paused, speed=speed, steps_per_second=rate, trial=trial)
                last_publish = now
            self.stop.wait(max(0, 1 / 30 - (time.monotonic() - tick_start)))
        self.save_checkpoint(sim, save_path)

    @staticmethod
    def save_checkpoint(sim, path):
        model = sim.export_model()
        if model:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(model))
            temp.replace(path)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/api/config":
            config = getattr(self.server.trainer, 'config', None)
            self.send_json(config if config is not None else {"error": "Not found"}, 200 if config is not None else 404)
            return
        if route == "/api/records" or route.startswith("/api/records/"):
            root = getattr(self.server.trainer, 'records_dir', None)
            if root is None:
                self.send_json([] if route == "/api/records" else {"error":"Not found"}, 200 if route == "/api/records" else 404)
                return
            if route == "/api/records":
                records = []
                for file in sorted(root.glob('*/manifest.json'), reverse=True):
                    try:
                        record = json.loads(file.read_text())
                        records.append({k:record[k] for k in ('id','seed','winner','runtime','holdout')})
                    except (ValueError, KeyError, OSError): continue
                self.send_json(records)
                return
            parts = route.split('/')
            if len(parts)!=5 or not parts[3].isascii() or not parts[3].isdigit() or parts[4] not in ('model','holdout'):
                self.send_json({"error":"Not found"},404)
                return
            filename = 'grand-prix.json' if parts[4]=='model' else 'grand-prix-holdout.json'
            file = root/parts[3]/filename
            if not file.is_file():
                self.send_json({"error":"Not found"},404)
                return
            self.send_json(json.loads(file.read_text()))
            return
        if route in ("/api/state", "/api/model"):
            with self.server.trainer.lock:
                data = self.server.trainer.state if route == "/api/state" else self.server.trainer.model
            self.send_json(data)
            return
        # Only distribution files are served; checkpoints and source stay private.
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/control":
            self.send_json({"error": "Not found"}, 404)
            return
        # Accept controls from this same-origin page, not arbitrary other websites.
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self.send_json({"error": "Origin mismatch"}, 403)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 100000: raise ValueError("请求大小无效")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json": raise ValueError("需要 JSON 请求")
            command = json.loads(self.rfile.read(size))
            if not isinstance(command, dict): raise ValueError("无效命令")
            if self.server.trainer.error: raise ValueError("训练器已停止：" + self.server.trainer.error)
            done, result = threading.Event(), {}
            self.server.trainer.commands.put_nowait((command, done, result))
            if not done.wait(15):
                self.send_json({"error": "训练器响应超时，请刷新检查状态"}, 503)
                return
            self.send_json(result, 400 if "error" in result else 200)
        except (ValueError, TypeError, queue.Full) as exc:
            self.send_json({"error": str(exc)}, 400)

    def send_json(self, data, status=200):
        body = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/state"): super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="GRID / EVO · 五站 F1 神经进化大奖赛")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--population", default=20, type=int)
    parser.add_argument("--candidates", default=32, type=int, help="每车手独立训练种群（8–64）")
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--checkpoint", default=str(Path(__file__).parent / "checkpoints/grand-prix.json"))
    args = parser.parse_args()
    if not 20 <= args.population <= 4096: parser.error("population must be 20..4096")
    device_info = cuda_environment()
    print("CUDA 自检通过：" + json.dumps(device_info, ensure_ascii=False), flush=True)
    if not 8 <= args.candidates <= 64: parser.error("candidates must be 8..64")
    from grand_prix import GrandPrixTrainer
    trainer = GrandPrixTrainer(args, device_info)
    handler = partial(Handler, directory=str(Path(__file__).parent / "dist"))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        server.trainer = trainer
        worker = threading.Thread(target=trainer.run, name="cuda-trainer", daemon=True)
        worker.start()
        print(f"EVO / 20 · http://{args.host}:{args.port} · Ctrl+C 停止", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            trainer.stop.set()
            worker.join(timeout=20)


if __name__ == "__main__":
    main()
