"""Batched neural evolution on CUDA. No browser-side training."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from race_common import WEIGHTS, DT, POINTS, WIDTH, HEIGHT, geometry, validate_model


@dataclass
class RunInfo:
    generation: int = 1
    elapsed: float = 0
    record: float = 0
    last_best: float = 0


class CudaEvolution:
    """All networks, physics, sensors, selection and mutations use device tensors.

    ``device='cpu'`` is only exposed for developer regression tests; the server
    requires a real CUDA device and never automatically falls back to CPU.
    """

    @torch.no_grad()
    def __init__(self, *, kind="flow", population=20, mutation=.16, seed=42, device="cuda", trial=False):
        if not (population == 1 if trial else 20 <= population <= 4096):
            raise ValueError("种群必须在 20 到 4096 之间")
        self.trial = trial
        self.last_end_reason = ""
        self.device = torch.device(device)
        self.population = population
        self.mutation = mutation
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.info = RunInfo()
        self.history = []
        self.champion = None
        self.best_path = []
        self._set_track(kind)
        self.weights = self._random((population, WEIGHTS)) * 1.6 - .8
        self._spawn()

    def _random(self, shape):
        return torch.rand(shape, device=self.device, generator=self.rng)

    def _set_track(self, kind):
        self.kind = kind
        points, mask, self.road_width = geometry(kind)
        self.points = torch.as_tensor(points, device=self.device)
        self.mask = torch.as_tensor(mask, device=self.device)
        self.ray_angles = torch.tensor([-1.35, -.9, -.45, 0, .45, .9, 1.35], device=self.device)
        self.ray_distances = torch.arange(0, 150, 4, device=self.device, dtype=torch.float32)
        self.offsets = torch.arange(-15, 16, device=self.device)

    def _spawn(self):
        n, d = self.population, self.device
        self.xy = self.points[0].repeat(n, 1)
        tangent = self.points[1] - self.points[0]
        self.angle = torch.atan2(tangent[1], tangent[0]).repeat(n)
        self.speed = torch.full((n,), 38., device=d)
        self.alive = torch.ones(n, dtype=torch.bool, device=d)
        self.stop_reasons = torch.zeros(n, dtype=torch.int8, device=d)
        self.index = torch.zeros(n, dtype=torch.long, device=d)
        self.progress = torch.zeros(n, device=d)
        self.best = torch.zeros(n, device=d)
        self.idle = torch.zeros(n, device=d)
        self.rays = torch.zeros((n, 7), device=d)
        self.inputs = torch.zeros((n, 8), device=d)
        self.hidden = torch.zeros((n, 10), device=d)
        self.outputs = torch.zeros((n, 2), device=d)
        # Continuous trials do not retain an ever-growing trajectory.
        self.trajectory = torch.empty((1 if self.trial else 1201, n, 2), device=d)
        self.trajectory[0] = self.xy
        self.ticks = 0
        self.info.elapsed = 0

    def _inside(self, xy):
        ix = torch.floor(xy[..., 0] / 3).long()
        iy = torch.floor(xy[..., 1] / 3).long()
        valid = (xy[..., 0] >= 0) & (xy[..., 0] < WIDTH) & (xy[..., 1] >= 0) & (xy[..., 1] < HEIGHT)
        return valid & self.mask[iy.clamp(0, self.mask.shape[0] - 1), ix.clamp(0, self.mask.shape[1] - 1)]

    @torch.no_grad()
    def step(self):
        angle = self.angle[:, None] + self.ray_angles[None, :]
        directions = torch.stack((angle.cos(), angle.sin()), dim=-1)
        probes = self.xy[:, None, None, :] + directions[:, :, None, :] * self.ray_distances[None, None, :, None]
        ray_hits = self._inside(probes)
        self.rays = torch.where(ray_hits, 150., self.ray_distances).amin(dim=-1)
        self.inputs = torch.cat((self.rays / 150, self.speed[:, None] / 155), dim=1)
        first = self.weights[:, :90].reshape(self.population, 10, 9)
        second = self.weights[:, 90:].reshape(self.population, 2, 11)
        self.hidden = torch.tanh(torch.bmm(first[:, :, :8], self.inputs.unsqueeze(-1)).squeeze(-1) + first[:, :, 8])
        self.outputs = torch.tanh(torch.bmm(second[:, :, :10], self.hidden.unsqueeze(-1)).squeeze(-1) + second[:, :, 10])
        moving = self.alive
        target = 38 + (self.outputs[:, 1] + 1) * .5 * 117
        self.speed = torch.where(moving, self.speed + (target - self.speed) * DT * 2, self.speed)
        self.angle = self.angle + self.outputs[:, 0] * self.speed / 45 * DT * moving
        direction = torch.stack((self.angle.cos(), self.angle.sin()), dim=-1)
        self.xy = self.xy + direction * self.speed[:, None] * DT * moving[:, None]
        candidates = (self.index[:, None] + self.offsets[None, :]) % POINTS
        distances = (self.points[candidates] - self.xy[:, None, :]).square().sum(dim=-1)
        next_index = candidates.gather(1, distances.argmin(dim=1, keepdim=True)).squeeze(1)
        delta = (next_index - self.index + POINTS // 2) % POINTS - POINTS // 2
        self.progress += delta * moving
        self.index = torch.where(moving, next_index, self.index)
        self.idle = torch.where(self.progress > self.best, 0., self.idle + DT)
        self.best = torch.maximum(self.best, self.progress)
        reasons = torch.where(~self._inside(self.xy), 1,
                              torch.where(self.idle > 4, 2,
                                          torch.where(self.progress < -12, 3, 0))).to(torch.int8)
        self.stop_reasons = torch.where(moving, reasons, self.stop_reasons)
        self.alive &= reasons == 0
        self.ticks += 1
        self.info.elapsed = self.ticks * DT
        if not self.trial:
            self.trajectory[self.ticks] = self.xy

    def finished(self):
        if self.trial:
            return not bool(self.alive.any().item())
        return self.ticks >= 1200 or bool((~self.alive.any() | (self.best.max() >= 840)).item())

    def end_reason(self):
        if self.trial:
            return {0: "", 1: "驶出有效路面", 2: "连续 4 秒未刷新最远进度", 3: "倒退越过起点"}[int(self.stop_reasons[0].item())]
        if self.ticks >= 1200:
            return "达到 40 秒评估时限"
        if self.best.max().item() >= 840:
            return "已有车辆完成两圈评估"
        return "本代车辆全部淘汰"

    def _update_champion(self):
        index = int(self.best.argmax().item())
        score = float(self.best[index].item())
        if self.trial:
            return index, score
        if score >= self.info.record and (score > 0 or self.champion is None):
            self.info.record = score
            self.champion = self.weights[index].clone()
            self.best_path = self.trajectory[:self.ticks + 1:3, index].cpu().tolist()
        return index, score

    @torch.no_grad()
    def next_generation(self, reason="手动结束本代"):
        if self.trial:
            raise ValueError("冠军试跑不进行繁衍")
        self._update_champion()
        self.last_end_reason = reason
        order = self.best.argsort(descending=True)
        self.info.last_best = float(self.best[order[0]].item())
        self.history.append({"generation": self.info.generation, "best": self.info.last_best, "mean": float(self.best.mean().item())})
        self.history = self.history[-200:]
        count = self.population - 3
        pool = order[:6]
        # Tournament selection, uniform crossover and mutations stay on CUDA.
        contestants = torch.randint(0, 6, (2, count, 2), generator=self.rng, device=self.device)
        candidates = pool[contestants]
        winners = candidates.gather(2, self.best[candidates].argmax(dim=2, keepdim=True)).squeeze(-1)
        a, b = self.weights[winners[0]], self.weights[winners[1]]
        children = torch.where(self._random(a.shape) < .18, b, a)
        noise = (self._random(a.shape) + self._random(a.shape) + self._random(a.shape) - 1.5) * .65
        children = (children + noise * (self._random(a.shape) < self.mutation)).clamp(-4, 4)
        elite = self.champion if self.champion is not None else self.weights[order[0]]
        self.weights = torch.cat((elite[None], self.weights[order[1]][None], children, self._random((1, WEIGHTS)) * 1.6 - .8), dim=0)
        self.info.generation += 1
        self._spawn()

    def export_model(self):
        self._update_champion()
        if self.champion is None:
            return None
        return {"format": "evo-racer-cuda-v1", "track": self.kind, "weights": self.champion.cpu().tolist(), "record": self.info.record}

    @torch.no_grad()
    def import_model(self, data):
        data = validate_model(data)
        self._set_track(data["track"])
        self.champion = torch.tensor(data["weights"], device=self.device, dtype=torch.float32)
        self.weights = (self.champion[None].repeat(self.population, 1) + (self._random((self.population, WEIGHTS)) - .5) * .25).clamp(-4, 4)
        self.weights[0] = self.champion
        self.info = RunInfo(record=data["record"])
        self.history, self.best_path = [], []
        self.last_end_reason = ""
        self._spawn()

    def snapshot(self):
        leader, _ = self._update_champion()
        # Only display data crosses to CPU; no browser data enters policy inputs.
        visible = list(range(min(20, self.population)))
        if leader not in visible:
            visible[-1] = leader
        packed = torch.cat((self.xy, self.angle[:, None], self.speed[:, None], self.alive[:, None], self.best[:, None], self.rays, self.inputs, self.hidden, self.outputs), dim=1)[visible].cpu().tolist()
        leader_weights = self.weights[leader].cpu().tolist()
        cars = []
        for i, row in zip(visible, packed):
            cars.append({"id": i, "x": row[0], "y": row[1], "angle": row[2], "speed": row[3], "alive": bool(row[4]), "fitness": row[5], "rays": row[6:13], "inputs": row[13:21], "hidden": row[21:31], "output": row[31:33], "weights": leader_weights if i == leader else []})
        return {"generation": self.info.generation, "elapsed": self.info.elapsed, "limit": None if self.trial else 40, "record": self.info.record, "lastBest": self.info.last_best, "history": self.history, "bestPath": [{"x": x, "y": y} for x, y in self.best_path], "kind": self.kind, "cars": cars, "leaderId": leader, "alive": int(self.alive.sum().item()), "population": self.population, "lastEndReason": self.last_end_reason}
