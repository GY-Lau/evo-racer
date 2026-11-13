"""Independent F1-labelled evolutionary populations and audited five-circuit trials."""
from __future__ import annotations
import hashlib
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from f1_config import DRIVERS, TRACKS, WIDTH, HEIGHT, POINTS, CELL, WIDTH_METERS, CAR_LENGTH, CAR_WIDTH, WHEELBASE, GEOMETRY_VERSION, HOLDOUT_INDEX, HOLDOUT_TRACK, circuit, public_config

DT, PARAMS, DRIVER_COUNT = 1 / 30, 112, len(DRIVERS)
VALIDATION_INTERVAL = 5
REASONS = ('行驶中', '驶出有效路面', '连续 6 秒无前进进度', '倒退越过起点', '完成目标圈数', '本轮训练评估结束', '本站免跑：此前验证未通过')


PHYSICS_VERSION = 4
# Reference tuning for a planar racing car, not measured team-specific F1 data.
GRAVITY, BASE_GRIP, DOWNFORCE_GRIP = 9.81, 1.55, .0032
ENGINE_ACCEL, BRAKE_ACCEL, DRAG = 8., 18., .0008
STEER_LIMIT, STEER_RATE, YAW_RESPONSE = math.radians(20), 1.8, .16


class Arena:
    """Device-resident simulation with graph replay and periodic survivor compaction."""
    STATE_NAMES = ('xy', 'angle', 'speed', 'index', 'progress', 'best', 'idle', 'reason',
                   'finish_tick', 'lap_ticks', 'yaw_rate', 'steer', 'long_accel',
                   'lateral_accel', 'grip_usage', 'throttle', 'brake',
                   'rays', 'inputs', 'hidden', 'outputs')

    @torch.no_grad()
    def __init__(self, weights, track, laps, assets, *, optimized=True):
        self.weights = weights.reshape(-1, PARAMS)
        self.device = weights.device
        self.track, self.laps = track, laps
        self.points, self.mask, self.scale, self.length = assets[track]
        n, d = len(self.weights), self.device
        self.xy = self.points[0].repeat(n, 1)
        tangent = self.points[1] - self.points[0]
        self.angle = torch.atan2(tangent[1], tangent[0]).repeat(n)
        self.speed = torch.full((n,), 30., device=d)
        self.index = torch.zeros(n, device=d, dtype=torch.long)
        for name in ('progress','best','idle','yaw_rate','steer','long_accel','lateral_accel','grip_usage','throttle','brake'):
            setattr(self, name, torch.zeros(n, device=d))
        self.reason = torch.zeros(n, device=d, dtype=torch.int8)
        self.finish_tick = torch.full((n,), -1, device=d, dtype=torch.long)
        self.lap_ticks = torch.full((n, laps), -1, device=d, dtype=torch.long)
        self.rays = torch.zeros((n, 7), device=d)
        self.inputs = torch.zeros((n, 8), device=d)
        self.hidden = torch.zeros((n, 10), device=d)
        self.outputs = torch.zeros((n, 2), device=d)
        self.offsets = torch.arange(-16, 17, device=d)
        self.ray_angles = torch.tensor([-1.4, -.9, -.45, 0, .45, .9, 1.4], device=d)
        self.distances = torch.arange(0, 182, 2, device=d, dtype=torch.float32)
        self.body_corners = torch.tensor([[CAR_LENGTH/2,CAR_WIDTH/2], [CAR_LENGTH/2,-CAR_WIDTH/2], [-CAR_LENGTH/2,CAR_WIDTH/2], [-CAR_LENGTH/2,-CAR_WIDTH/2]], device=d)
        self.ticks = 0
        self.tick_tensor = torch.zeros((), device=d, dtype=torch.long)
        self.optimized = optimized and d.type == 'cuda'
        self.graph = None
        self.active_indices = torch.arange(n, device=d)
        self.active_count = n

    def inside(self, xy):
        ix, iy = (xy[..., 0] / CELL).floor().long(), (xy[..., 1] / CELL).floor().long()
        valid = (xy[..., 0] >= 0) & (xy[..., 0] < WIDTH) & (xy[..., 1] >= 0) & (xy[..., 1] < HEIGHT)
        return valid & self.mask[iy.clamp(0, self.mask.shape[0] - 1), ix.clamp(0, self.mask.shape[1] - 1)]

    # 静态成员函数，类似 C++ 的 static 方法，不需要通过 self 访问对象成员。
    @staticmethod
    def dynamics(speed, yaw_rate, steer, outputs):
        """根据网络指令，计算一辆或一批赛车在下一个时间步的运动状态。

        输入符号及单位：
            speed = v：当前车速，m/s；显示为 km/h 时需要乘以 3.6。
            yaw_rate = r：当前车身航向角变化率，rad/s，不是方向盘转角。
            steer = δ：当前等效前轮转角，rad，用于简化自行车模型。
            outputs：形状为 [车辆数, 2] 的网络输出，两个值均在 [-1, 1] 内。
                outputs[:, 0] 是所有车辆的转向指令；冒号表示取全部行。
                outputs[:, 1] 是所有车辆的目标速度指令，不是直接的油门开度。

        时间步 Δt = DT = 1/30 秒，等效轴距 L = WHEELBASE = 3.4 米。
        下方公式逐辆成立，PyTorch 张量运算会同时处理整批车辆。
        clamp(x, a, b) 表示限幅，类似 C++ 的 std::clamp(x, a, b)。
        这些是本模拟器的参考参数和近似模型，不是某辆真实 F1 赛车的测量数据。
        """
        # ① 将归一化转向指令映射成目标前轮转角：δ_cmd = u_steer × δ_max。
        # STEER_LIMIT = 20°，已转换为弧度，因此目标范围为 [-20°, +20°]。
        command = outputs[:,0] * STEER_LIMIT

        # ② 限制前轮每一步能转动多少，避免瞬间从左满舵跳到右满舵。
        # δ_next = δ + clamp(δ_cmd - δ, -δ_rate × Δt, +δ_rate × Δt)。
        # STEER_RATE = 1.8 rad/s，所以每步最多改变 0.06 rad，约 3.44°。
        steer = steer + (command-steer).clamp(-STEER_RATE*DT, STEER_RATE*DT)

        # ③ 将另一个网络输出映射成目标速度：v_target = 8 + (u_speed + 1) × 43.5。
        # 输出 -1 对应 8 m/s，输出 +1 对应 95 m/s，即约 28.8～342 km/h。
        # 这是控制器希望达到的速度，车辆实际速度仍受后续加减速过程影响。
        target = 8 + (outputs[:,1]+1)*43.5

        # ④ 用比例控制器将速度误差转换成纵向加速度指令，并限制执行能力。
        # a_drive = clamp(Kp × (v_target - v), -a_brake_max, +a_engine_max)。
        # Kp = 2.5 /s；驱动上限 8 m/s²，制动指令下限 -18 m/s²。
        # drive 是加速度指令，不是力，也不是最终显示的油门百分比。
        drive = ((target-speed)*2.5).clamp(-BRAKE_ACCEL, ENGINE_ACCEL)

        # ⑤ 估计当前可用的合成加速度预算，以加速度表示抓地能力：
        # a_grip = μ_reference × g + k_downforce × v²。
        # 本模型取 μ_reference = 1.55、g = 9.81 m/s²、k_downforce = 0.0032 /m。
        # 速度平方项近似表示下压力增加的抓地能力，不是完整空气动力学计算。
        grip = GRAVITY*BASE_GRIP + DOWNFORCE_GRIP*speed.square()

        # ⑥ 扣除速度平方阻力，再限制净纵向加速度：
        # a_x = clamp(a_drive - k_drag × v², -a_grip, +a_grip)。
        # DRAG = 0.0008 /m；speed.square() 就是逐元素计算 v × v。
        # 阻力已作为加速度项建模，因此这里没有再除以车辆质量。
        acceleration = (drive-DRAG*speed.square()).clamp(-grip, grip)

        # ⑦ 用当前净加速度更新车速：v_next = max(v + a_x × Δt, 0)。
        # clamp_min(0) 防止出现负车速；这个简化模型不模拟挂倒挡。
        velocity = (speed+acceleration*DT).clamp_min(0)

        # ⑧ 按摩擦圆近似，把合成抓地预算分给纵向和横向：
        # a_x² + a_y² ≤ a_grip²，所以 a_y_available = sqrt(max(a_grip² - a_x², 0))。
        # 本实现优先分配纵向加减速，剩余预算才用于转弯。
        # 简化之处：这里把包含空气阻力的净 a_x 也计入预算；真实轮胎摩擦圆
        # 应按轮胎产生的纵向力计算，空气阻力本身不直接占用轮胎抓地力。
        lateral_budget = (grip.square()-acceleration.square()).clamp_min(0).sqrt()

        # ⑨ 根据简化自行车几何，计算这个前轮转角所要求的航向角速度：
        # r_requested = v_next / L × tan(δ_next)。
        # 对应转弯半径 R = L / tan(δ_next)，并使用 r = v / R。
        # 此时只是理想请求，尚未检查抓地力是否允许实际达到。
        requested = velocity/WHEELBASE * torch.tan(steer)

        # ⑩ 用一阶滞后近似车身转向响应：
        # r_candidate = r + α × (r_requested - r)，α = Δt / (τ + Δt)。
        # YAW_RESPONSE = τ = 0.16 秒，α 约为 0.172；车身不会立刻达到请求值。
        # 这是一阶响应近似，没有求解真实轮胎侧偏角和车身横向速度。
        yaw = yaw_rate + (requested-yaw_rate) * (DT/(YAW_RESPONSE+DT))

        # ⑪ 根据横向加速度关系 a_y = v × r，计算允许的航向角速度上限：
        # r_limit = a_y_available / max(v_next, 2 m/s)。
        # 分母设置低速下限，避免除以零，也避免低速时产生异常大的角速度上限。
        limit = lateral_budget/velocity.clamp_min(2)

        # ⑫ 将实际航向角速度限制在抓地预算内：
        # r_next = clamp(r_candidate, -r_limit, +r_limit)。
        # 请求超限时，实际转弯能力不足、转弯半径变大，用来近似高速推头；
        # 这里限制的是转弯能力，没有把车速瞬间降低到“安全弯速”。
        yaw = torch.maximum(torch.minimum(yaw,limit),-limit)

        # ⑬ 计算实际横向加速度：a_y = r_next × v_next，单位 m/s²。
        # 页面横向 G 值由 a_y / 9.81 得到；正负号表示不同转弯方向。
        lateral = yaw*velocity

        # ⑭ 合成加速度占抓地预算的比例：usage = sqrt(a_x² + a_y²) / a_grip。
        # 0 表示未使用，1 表示用满本模型的预算；它不是轮胎滑移率。
        usage = (acceleration.square()+lateral.square()).sqrt()/grip

        # ⑮ 将正的加速度指令归一化，作为显示和遥测用的油门开度：
        # throttle = clamp(a_drive / a_engine_max, 0, 1)。
        throttle = (drive/ENGINE_ACCEL).clamp(0,1)

        # ⑯ 将负的加速度指令归一化，作为显示和遥测用的制动开度：
        # brake = clamp(-a_drive / a_brake_max, 0, 1)。
        # 它与 throttle 都从 drive 派生，不是在这里再次施加一遍加减速。
        brake = (-drive/BRAKE_ACCEL).clamp(0,1)

        # 返回：新车速、新航向角速度、新前轮转角、纵向/横向加速度、预算使用率、
        # 油门开度、制动开度。车身航向角和 x/y 位置的积分更新在 _compute() 中。
        return velocity,yaw,steer,acceleration,lateral,usage,throttle,brake

    def _compute(self):
        ids = self.active_indices
        # Indices and tensor addresses stay fixed for one captured graph.
        batch = {name: getattr(self,name).index_select(0,ids) for name in self.STATE_NAMES}
        xy,angle,speed = batch['xy'],batch['angle'],batch['speed']
        moving = batch['reason']==0
        a = angle[:,None]+self.ray_angles[None]
        direction = torch.stack((a.cos(),a.sin()),dim=-1)
        probes = xy[:,None,None]+direction[:,:,None]*(self.distances*self.scale)[None,None,:,None]
        rays = torch.where(self.inside(probes),180.,self.distances).amin(dim=-1)
        inputs = torch.cat((rays/180,speed[:,None]/95),dim=1)
        weights = self.weights.index_select(0,ids)
        first,second = weights[:,:90].reshape(-1,10,9),weights[:,90:].reshape(-1,2,11)
        hidden = torch.tanh((first[:,:,:8]*inputs[:,None,:]).sum(-1)+first[:,:,8])
        outputs = torch.tanh((second[:,:,:10]*hidden[:,None,:]).sum(-1)+second[:,:,10])
        dynamic = self.dynamics(speed,batch['yaw_rate'],batch['steer'],outputs)
        speed,yaw_rate,steer,ax,ay,usage,throttle,brake = dynamic
        speed = torch.where(moving,speed,batch['speed'])
        angle = angle+yaw_rate*DT*moving
        xy = xy+torch.stack((angle.cos(),angle.sin()),dim=-1)*(speed*self.scale*DT*moving)[:,None]
        candidates = (batch['index'][:,None]+self.offsets[None])%POINTS
        starts = self.points[candidates]
        segments = self.points[(candidates+1)%POINTS]-starts
        relative = xy[:,None]-starts
        projection = ((relative*segments).sum(-1)/segments.square().sum(-1).clamp_min(1e-8)).clamp(0,1)
        distances = (relative-projection[...,None]*segments).square().sum(-1)
        nearest = distances.argmin(dim=1)
        along = projection.gather(1,nearest[:,None]).squeeze(1)
        index = (candidates.gather(1,nearest[:,None]).squeeze(1)+(along>.5).long())%POINTS
        delta = (index-batch['index']+POINTS//2)%POINTS-POINTS//2
        progress = batch['progress']+delta*moving
        index = torch.where(moving,index,batch['index'])
        idle = torch.where(progress>batch['best'],0.,batch['idle']+DT)
        best = torch.maximum(batch['best'],progress)
        forward = torch.stack((angle.cos(),angle.sin()),dim=-1)
        side = torch.stack((-angle.sin(),angle.cos()),dim=-1)
        corners = xy[:,None]+(forward[:,None]*self.body_corners[None,:,0,None]+side[:,None]*self.body_corners[None,:,1,None])*self.scale
        corner_relative = corners[:,:,None]-starts[:,None]
        corner_projection = ((corner_relative*segments[:,None]).sum(-1)/segments.square().sum(-1)[:,None].clamp_min(1e-8)).clamp(0,1)
        corner_distance = (corner_relative-corner_projection[...,None]*segments[:,None]).square().sum(-1).amin(-1)
        on_road = (corner_distance <= (WIDTH_METERS[self.track]*self.scale/2)**2).all(dim=1)
        reason = torch.where(~on_road,1,torch.where(idle>6,2,torch.where(progress < -12,3,0)))
        completed = (progress >= POINTS*self.laps)&(reason==0)
        reason = torch.where(completed,4,reason).to(torch.int8)
        reason = torch.where(moving,reason,batch['reason'])
        self.tick_tensor.add_(1)
        finish_tick = torch.where(moving&completed,self.tick_tensor,batch['finish_tick'])
        lap_ticks = batch['lap_ticks']
        for lap in range(self.laps):
            crossed = moving&on_road&(progress>=POINTS*(lap+1))&(lap_ticks[:,lap]<0)
            lap_ticks[:,lap] = torch.where(crossed,self.tick_tensor,lap_ticks[:,lap])
        values = dict(xy=xy,angle=angle,speed=speed,index=index,progress=progress,best=best,idle=idle,
            reason=reason,finish_tick=finish_tick,lap_ticks=lap_ticks,yaw_rate=yaw_rate,steer=steer,
            long_accel=ax,lateral_accel=ay,grip_usage=usage,throttle=throttle,brake=brake,
            rays=rays,inputs=inputs,hidden=hidden,outputs=outputs)
        for name,value in values.items():
            ref = getattr(self,name)
            mask = moving.reshape((-1,)+(1,)*(value.ndim-1))
            ref.index_copy_(0,ids,torch.where(mask,value,batch[name]))

    def _capture(self, alive):
        self.graph = None
        self.active_indices = torch.as_tensor(alive,device=self.device,dtype=torch.long)
        self.active_count = len(alive)
        if not self.active_count: return
        backups = {name:getattr(self,name).clone() for name in self.STATE_NAMES}
        tick = self.tick_tensor.clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2): self._compute()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        for name,value in backups.items(): getattr(self,name).copy_(value)
        self.tick_tensor.copy_(tick)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): self._compute()
        # Capture and warmup must never advance the real attempt.
        for name,value in backups.items(): getattr(self,name).copy_(value)
        self.tick_tensor.copy_(tick)
        self.graph = graph

    @torch.no_grad()
    def step(self):
        if self.optimized:
            if self.graph is None or self.ticks%32==0:
                alive = np.flatnonzero(self.reason.cpu().numpy()==0)
                if self.graph is None or len(alive)<=self.active_count//2:
                    self._capture(alive)
            if self.graph is not None: self.graph.replay()
            else: self.tick_tensor.add_(1)
        else:
            self._compute()
        self.ticks += 1

    def done(self):
        return bool((self.reason!=0).all().item())

    def save(self):
        return dict(ticks=self.ticks,tensors={name:getattr(self,name).cpu().tolist() for name in self.STATE_NAMES})

    def restore(self, data):
        ticks = data['ticks']
        if type(ticks) is not int or not 0<=ticks<100000000:
            raise ValueError('无效模拟计时')
        for name,values in data['tensors'].items():
            if name not in self.STATE_NAMES: raise ValueError('无效模拟字段')
            ref = getattr(self,name)
            array = np.asarray(values)
            if array.shape!=tuple(ref.shape) or not np.isfinite(array).all():
                raise ValueError('无效模拟张量')
            ref.copy_(torch.as_tensor(array,dtype=ref.dtype,device=self.device))
        self.ticks = ticks
        self.tick_tensor.fill_(ticks)
        self.graph = None


class GrandPrix:
    @torch.no_grad()
    def __init__(self, *, candidates=32, seed=2026, device='cuda'):
        if not 8 <= candidates <= 64:
            raise ValueError('每车手训练种群须为 8–64')
        self.device, self.candidates, self.seed = torch.device(device), candidates, seed
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.assets = []
        for i in range(5):
            points, mask, scale, length = circuit(i)
            self.assets.append((torch.as_tensor(points, device=self.device), torch.as_tensor(mask, device=self.device), scale, length))
        self.weights = self.random((DRIVER_COUNT, candidates, PARAMS)) * 2 - 1
        self.champions = self.weights[:, 0].clone()
        self.scores = torch.zeros((DRIVER_COUNT, candidates, 5), device=self.device)
        self.champion_scores = torch.zeros((DRIVER_COUNT, 5), device=self.device)
        self.results = torch.zeros((DRIVER_COUNT, 5), device=self.device)
        self.pass_ticks = torch.full((DRIVER_COUNT, 5), -1, device=self.device, dtype=torch.long)
        self.validation_skipped = np.zeros((DRIVER_COUNT,5),dtype=bool)
        self.model_hashes = self.hashes()
        self.generation, self.total_steps, self.stage = 1, 0, 0
        self.next_validation_generation = VALIDATION_INTERVAL
        self.phase, self.mutation = 'training', .16
        self.winner = None
        self.history, self.events = [], []
        self.announced = set()
        self.arena = Arena(self.weights, 0, 1, self.assets)
        self.log('LIGHTS OUT', '五站训练大奖赛开始，22 位车手从随机大脑出发。')

    def random(self, shape):
        return torch.rand(shape, device=self.device, generator=self.rng)

    def hashes(self):
        return [hashlib.sha256(w.astype('<f4').tobytes()).hexdigest() for w in self.champions.cpu().numpy()]

    def log(self, tag, text):
        self.events.insert(0, dict(id=self.total_steps, tag=tag, text=text, generation=self.generation))
        self.events = self.events[:12]

    @staticmethod
    def fitness(scores):
        progress = scores.clamp(0, 1)
        return progress.amin(dim=-1) * 4 + progress.mean(dim=-1) + (scores - 1).clamp_min(0).mean(dim=-1) * .05

    @torch.no_grad()
    def evolve(self):
        if self.phase == 'validation':
            self.next_validation_generation = self.generation + VALIDATION_INTERVAL
        order = self.fitness(self.scores).argsort(dim=1, descending=True)
        rows = torch.arange(DRIVER_COUNT, device=self.device)[:, None]
        pool = self.weights[rows, order[:, :6]]
        n = self.candidates - 3
        a_index = torch.randint(0, 6, (DRIVER_COUNT, n), generator=self.rng, device=self.device)
        b_index = torch.randint(0, 6, (DRIVER_COUNT, n), generator=self.rng, device=self.device)
        a, b = pool[rows, a_index], pool[rows, b_index]
        children = torch.where(self.random(a.shape) < .5, a, b)
        noise = torch.randn(children.shape, device=self.device, generator=self.rng) * .22
        children = (children + noise * (self.random(children.shape) < self.mutation)).clamp(-4, 4)
        # Each row is one driver's population. No cross-driver gene transfer.
        self.weights = torch.cat((self.champions[:, None], pool[:, 1:2], children,
                                  self.random((DRIVER_COUNT, 1, PARAMS)) * 2 - 1), dim=1)
        self.scores.zero_()
        self.generation += 1
        self.phase, self.stage = 'training', 0
        self.arena = Arena(self.weights, 0, 1, self.assets)
        self.log('EVOLUTION', f'第 {self.generation} 轮开始：各车手独立选择、交叉和变异。')

    @torch.no_grad()
    def step(self):
        if self.winner:
            return False
        self.arena.step()
        self.total_steps += 1
        if self.arena.ticks % 4:
            return False
        if self.phase == 'training':
            if self.arena.ticks >= 3600 or self.arena.done():
                self.finish_training_stage()
                return True
        else:
            self.update_validation()
            if self.winner:
                return True
            if self.arena.done():
                eligible=(self.pass_ticks[:,:self.stage+1]>=0).all(dim=1).cpu().numpy()
                self.validation_skipped[~eligible,self.stage+1:]=True
                if self.stage == 4 or not eligible.any():
                    if self.stage<4: self.log('EARLY EXIT','本轮所有模型均已验证失败，提前结束验证并继续训练。')
                    self.evolve()
                else:
                    self.stage += 1
                    self.arena = Arena(self.champions.clone(), self.stage, 3, self.assets)
                    self.arena.reason[torch.as_tensor(~eligible,device=self.device)]=6
                    self.log('VALIDATION', f'固定模型验证 · {TRACKS[self.stage]["short"]}站，每车连续 3 圈。')
                return True
        return False

    @torch.no_grad()
    def finish_training_stage(self):
        arena = self.arena
        progress = (arena.best / POINTS).clamp(0, 1)
        bonus = torch.where(arena.reason == 4, (1 - arena.finish_tick / 3600).clamp(0, 1) * .1, 0.)
        self.scores[:, :, self.stage] = (progress + bonus).reshape(DRIVER_COUNT, self.candidates)
        if self.stage < 4:
            self.stage += 1
            self.arena = Arena(self.weights, self.stage, 1, self.assets)
            self.log('PRACTICE', f'第 {self.generation} 轮 · {TRACKS[self.stage]["short"]}站训练。')
            return
        best = self.fitness(self.scores).argmax(dim=1)
        rows = torch.arange(DRIVER_COUNT, device=self.device)
        chosen = self.scores[rows, best]
        improve = self.fitness(chosen) >= self.fitness(self.champion_scores)
        self.champions = torch.where(improve[:, None], self.weights[rows, best], self.champions)
        self.champion_scores = torch.where(improve[:, None], chosen, self.champion_scores)
        self.model_hashes = self.hashes()
        self.history.append(dict(generation=self.generation, minimum=self.champion_scores.clamp(0, 1).amin(dim=1).cpu().tolist(),
                                 mean=self.champion_scores.clamp(0, 1).mean(dim=1).cpu().tolist()))
        self.history = self.history[-80:]
        # A new selected model must never inherit another model's validation.
        self.results.zero_()
        self.pass_ticks.fill_(-1)
        self.validation_skipped.fill(False)
        self.announced.clear()
        if self.generation < self.next_validation_generation:
            self.evolve()
            return
        self.phase, self.stage = 'validation', 0
        self.arena = Arena(self.champions.clone(), 0, 3, self.assets)
        self.log('PARC FERMÉ', '已完成 5 轮训练，各车手大脑冻结，开始五站连续 3 圈验证。')

    @torch.no_grad()
    def update_validation(self):
        self.results[:, self.stage] = (self.arena.best / POINTS).clamp(0, 3)
        completed = self.arena.reason == 4
        self.pass_ticks[:, self.stage] = torch.where(completed, self.arena.finish_tick, self.pass_ticks[:, self.stage])
        # One status copy replaces repeated nonzero/item synchronizations per driver.
        reasons=self.arena.reason.cpu().numpy()
        completed_ids=np.flatnonzero(reasons==4).tolist()
        newly_completed=[i for i in completed_ids if (i,self.stage) not in self.announced]
        for driver in newly_completed:
            key = (driver, self.stage)
            if key not in self.announced:
                self.announced.add(key)
                self.log('PASS', f'{DRIVERS[driver]["code"]} 通过{TRACKS[self.stage]["short"]}站 · 连续 3 圈。')
        if self.stage != 4 or not completed_ids:
            return
        eligible = (self.pass_ticks >= 0).all(dim=1).nonzero().flatten().cpu().tolist()
        if not eligible:
            return
        # Physics finish tick wins; total five-circuit time is an explicit tie-break.
        times = self.pass_ticks.cpu().tolist()
        keys = {i: (times[i][4], sum(times[i])) for i in eligible}
        first = min(keys.values())
        winners = [i for i in eligible if keys[i] == first]
        elapsed = self.total_steps - self.arena.ticks + first[0]
        self.winner = dict(drivers=winners, generation=self.generation, totalSteps=elapsed,
                           simulationSeconds=elapsed * DT, validationSeconds=first[1] * DT,
                           certificates=[dict(driver=i, modelHash=self.model_hashes[i],
                                              trackIds=[t['id'] for t in TRACKS], laps=[3]*5,
                                              ticks=times[i], weights=self.champions[i].cpu().tolist()) for i in winners])
        self.phase = 'finished'
        self.log('WORLD CHAMPION', ' / '.join(DRIVERS[i]['name'] for i in winners) + ' 首先通过全部五站验证！')

    def snapshot(self):
        arena = self.arena
        if self.phase == 'training':
            base = torch.arange(DRIVER_COUNT, device=self.device) * self.candidates
            active = (arena.reason == 0).reshape(DRIVER_COUNT, self.candidates)
            preferred = (arena.best.reshape(DRIVER_COUNT, self.candidates) + active * 100000).argmax(dim=1) + base
            prior = getattr(arena, 'display_indices', preferred)
            # Stay with the same physical model until its attempt ends.
            indices = torch.where(arena.reason[prior] == 0, prior, preferred)
            arena.display_indices = indices
        else:
            indices = torch.arange(DRIVER_COUNT, device=self.device)
        packed = torch.cat((arena.xy, arena.angle[:, None], arena.speed[:, None], arena.best[:, None] / POINTS,
                            arena.reason[:, None], arena.rays, arena.inputs, arena.hidden, arena.outputs), dim=1)[indices].cpu().numpy().astype(np.float64).round(3).tolist()
        networks = arena.weights[indices].cpu().numpy().astype(np.float64).round(4).tolist()
        identities = indices.cpu().tolist()
        cars = [dict(id=i, x=v[0], y=v[1], angle=v[2], speed=round(v[3]*3.6, 1), laps=v[4], reason=REASONS[int(v[5])],
                     candidate=identities[i], status=int(v[5]), rays=v[6:13], inputs=v[13:21], hidden=v[21:31], outputs=v[31:33], weights=networks[i]) for i, v in enumerate(packed)]
        telemetry=torch.stack((arena.long_accel,arena.lateral_accel,arena.grip_usage,arena.throttle,arena.brake,arena.steer),dim=1)[indices].cpu().tolist()
        for car,v in zip(cars,telemetry):
            car.update(longG=round(v[0]/GRAVITY,2),lateralG=round(v[1]/GRAVITY,2),gripUsage=round(v[2],3),throttle=round(v[3],2),brake=round(v[4],2),steer=round(v[5],3))
        scores = self.champion_scores.clamp(0, 1).cpu().tolist()
        results, ticks = self.results.cpu().tolist(), self.pass_ticks.cpu().tolist()
        drivers = [dict(id=i, passed=sum(t>=0 for t in ticks[i]), progress=scores[i], validation=results[i],
                        validationTimes=[round(t*DT, 2) if t>=0 else None for t in ticks[i]], validationSkipped=self.validation_skipped[i].tolist(),
                        minimum=min(scores[i]), average=sum(scores[i])/5, modelHash=self.model_hashes[i]) for i in range(DRIVER_COUNT)]
        ranking = sorted(range(DRIVER_COUNT), key=lambda i: (-drivers[i]['passed'], -drivers[i]['minimum'], -drivers[i]['average'], -cars[i]['laps'], i))
        return dict(ready=True, phase=self.phase, generation=self.generation, track=self.stage, elapsed=round(arena.ticks*DT, 2),
                    validationInterval=VALIDATION_INTERVAL,
                    nextValidationGeneration=self.generation if self.phase!='training' else self.next_validation_generation,
                    totalSteps=self.total_steps, simulationSeconds=round(self.total_steps*DT, 1), cars=cars, drivers=drivers, ranking=ranking,
                    active=int((arena.reason==0).sum().item()), networks=DRIVER_COUNT*self.candidates,
                    physicsVersion=PHYSICS_VERSION, execution='cuda-graph-survivors', candidates=self.candidates, mutation=self.mutation, goalLaps=1 if self.phase=='training' else 3,
                    history=self.history, events=list(self.events), winner=self.winner, trackScale=arena.scale)

    def save(self):
        return dict(format='evo-f1-grand-prix-v1', nextValidationGeneration=self.next_validation_generation, validationInterval=VALIDATION_INTERVAL, geometryVersion=GEOMETRY_VERSION, physicsVersion=PHYSICS_VERSION, candidates=self.candidates, seed=self.seed,
                    generation=self.generation, totalSteps=self.total_steps, stage=self.stage, phase=self.phase,
                    mutation=self.mutation, weights=self.weights.cpu().tolist(), champions=self.champions.cpu().tolist(),
                    scores=self.scores.cpu().tolist(), championScores=self.champion_scores.cpu().tolist(),
                    results=self.results.cpu().tolist(), passTicks=self.pass_ticks.cpu().tolist(), validationSkipped=self.validation_skipped.tolist(),
                    hashes=self.model_hashes, rng=self.rng.get_state().cpu().tolist(), arena=self.arena.save(),
                    history=self.history, events=self.events, winner=self.winner)

    @classmethod
    def restore(cls, data, device='cuda'):
        if not isinstance(data, dict) or data.get('format') != 'evo-f1-grand-prix-v1':
            raise ValueError('赛事存档格式不匹配')
        obj = cls(candidates=data['candidates'], seed=data['seed'], device=device)
        for name, key in [('weights','weights'), ('champions','champions'), ('scores','scores'), ('champion_scores','championScores'), ('results','results'), ('pass_ticks','passTicks')]:
            ref = getattr(obj, name)
            array = np.asarray(data[key])
            if array.shape != tuple(ref.shape) or not np.isfinite(array).all():
                raise ValueError('赛事存档张量无效：' + key)
            if name in ('weights', 'champions') and np.abs(array).max() > 4:
                raise ValueError('赛事权重超出范围')
            setattr(obj, name, torch.as_tensor(array, dtype=ref.dtype, device=obj.device))
        for name in ('generation','total_steps','stage','phase','mutation','history','events','winner'):
            key = {'total_steps':'totalSteps'}.get(name, name)
            setattr(obj, name, data[key])
        if obj.phase not in ('training','validation','finished') or type(obj.stage) is not int or not 0<=obj.stage<5:
            raise ValueError('赛事阶段无效')
        obj.next_validation_generation = data.get('nextValidationGeneration',obj.generation if obj.phase!='training' else obj.generation+VALIDATION_INTERVAL-1)
        skipped=np.asarray(data.get('validationSkipped',np.zeros((DRIVER_COUNT,5),dtype=bool)),dtype=bool)
        if skipped.shape!=(DRIVER_COUNT,5): raise ValueError('无效验证淘汰记录')
        obj.validation_skipped=skipped
        obj.model_hashes = obj.hashes()
        if obj.model_hashes != data['hashes']:
            raise ValueError('冠军模型校验失败')
        obj.rng.set_state(torch.tensor(data['rng'], dtype=torch.uint8, device='cpu'))
        obj.arena = Arena(obj.weights if obj.phase=='training' else obj.champions.clone(), obj.stage, 1 if obj.phase=='training' else 3, obj.assets)
        obj.arena.restore(data['arena'])
        obj.announced = {(i,j) for i,row in enumerate(obj.pass_ticks.cpu().tolist()) for j,t in enumerate(row) if t>=0}
        if data.get('geometryVersion',1)!=GEOMETRY_VERSION or data.get('physicsVersion',1)!=PHYSICS_VERSION:
            # Preserve learned populations, but never certify old geometry results.
            obj.weights[:, 0] = obj.champions
            obj.scores.zero_()
            obj.champion_scores.zero_()
            obj.results.zero_()
            obj.pass_ticks.fill_(-1)
            obj.validation_skipped.fill(False)
            obj.winner = None
            obj.phase, obj.stage = 'training', 0
            obj.generation += 1
            obj.next_validation_generation = obj.generation + VALIDATION_INTERVAL - 1
            obj.history = []
            obj.announced.clear()
            obj.arena = Arena(obj.weights, 0, 1, obj.assets)
            obj.log('CIRCUIT UPDATE', '赛道或车辆物理已更新；保留已学模型，重新评估五站，旧证书仅保留在归档。')
        return obj


class HoldoutTrial:
    """Inference-only, unseen-circuit trial of an audited winning certificate."""
    def __init__(self, game, driver=None):
        if not game.winner:
            raise ValueError('冠军产生后才能测试泛化')
        certificates = game.winner['certificates']
        certificate = next((c for c in certificates if driver is None or c['driver']==driver), None)
        if certificate is None:
            raise ValueError('该车手没有冠军证书')
        values = np.asarray(certificate['weights'], dtype='<f4')
        if values.shape != (PARAMS,) or not np.isfinite(values).all():
            raise ValueError('冠军模型权重无效')
        self.model_hash = hashlib.sha256(values.tobytes()).hexdigest()
        if self.model_hash != certificate['modelHash']:
            raise ValueError('冠军证书校验失败')
        self.driver = certificate['driver']
        self.started = time.time_ns()
        points, mask, scale, length = circuit(HOLDOUT_INDEX)
        assets = list(game.assets) + [(torch.as_tensor(points, device=game.device), torch.as_tensor(mask, device=game.device), scale, length)]
        weights = torch.tensor(values.copy(), device=game.device)
        self.arena = Arena(weights, HOLDOUT_INDEX, 3, assets)

    def step(self):
        if self.arena.done(): return False
        self.arena.step()
        return self.arena.done()

    def summary(self):
        a = self.arena
        status = int(a.reason[0].item())
        ticks = a.lap_ticks[0].cpu().tolist()
        times = [(t-(ticks[i-1] if i else 0))*DT if t>=0 else None for i,t in enumerate(ticks)]
        actual = hashlib.sha256(a.weights.cpu().numpy().astype('<f4').tobytes()).hexdigest()
        return dict(driver=self.driver, track=HOLDOUT_INDEX, trackId=HOLDOUT_TRACK['id'],
                    started=self.started, status='running' if status==0 else 'passed' if status==4 else 'failed',
                    reason=REASONS[status], laps=round(float(a.best[0].item())/POINTS,3), goalLaps=3,
                    elapsed=round(a.ticks*DT,2), lapSeconds=[round(t,2) if t is not None else None for t in times],
                    modelHash=self.model_hash, actualModelHash=actual, weightsUnchanged=actual==self.model_hash,
                    geometryVersion=GEOMETRY_VERSION, physicsVersion=PHYSICS_VERSION, trainedOnTrack=False)

    def snapshot(self, game):
        s = game.snapshot()
        a = self.arena
        v = torch.cat((a.xy, a.angle[:,None], a.speed[:,None], a.best[:,None]/POINTS,
                       a.reason[:,None], a.rays, a.inputs, a.hidden, a.outputs), dim=1)[0].cpu().tolist()
        for c in s['cars']:
            c.update(hiddenCar=True, status=1, reason='未参加泛化测试')
        s['cars'][self.driver] = dict(id=self.driver, x=v[0], y=v[1], angle=v[2], speed=round(v[3]*3.6,1), laps=v[4],
            status=int(v[5]), reason=REASONS[int(v[5])], candidate=0, hiddenCar=False, throttle=round(a.throttle[0].item(),2),brake=round(a.brake[0].item(),2),lateralG=round(a.lateral_accel[0].item()/GRAVITY,2),gripUsage=round(a.grip_usage[0].item(),3),rays=v[6:13], inputs=v[13:21],
            hidden=v[21:31], outputs=v[31:33], weights=a.weights[0].cpu().tolist())
        s.update(phase='holdout', track=HOLDOUT_INDEX, trackScale=a.scale, elapsed=round(a.ticks*DT,2),
                 totalSteps=game.total_steps+a.ticks, active=int(not a.done()), goalLaps=3, holdout=self.summary())
        return s

    def save(self, path):
        data = dict(format='evo-holdout-v1', result=self.summary(), arena=self.arena.save())
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, separators=(',',':'), allow_nan=False))
        temp.replace(path)

    @classmethod
    def restore(cls, game, data):
        result = data['result']
        if data.get('format')!='evo-holdout-v1' or result['geometryVersion']!=GEOMETRY_VERSION or result.get('physicsVersion',1)!=PHYSICS_VERSION or result['trackId']!=HOLDOUT_TRACK['id']:
            raise ValueError('泛化测试存档不匹配')
        obj = cls(game, result['driver'])
        if obj.model_hash != result['modelHash']: raise ValueError('泛化测试模型不匹配')
        obj.started = result['started']
        obj.arena.restore(data['arena'])
        return obj


class GrandPrixTrainer:
    def __init__(self, args, device_info):
        self.args, self.device_info = args, device_info
        self.records_dir = Path(args.checkpoint).parent.parent / "records"
        self.config = public_config(args.candidates)
        self.commands = queue.Queue(maxsize=32)
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.state, self.model, self.error = {'ready':False}, None, None

    def checkpoint(self, game, path, *, paused, speed, runtime):
        data = game.save()
        data['settings'] = dict(paused=paused, speed=speed, runtime=runtime)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, separators=(',',':'), allow_nan=False))
        temp.replace(path)

    def run(self):
        try:
            self._run()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = str(exc)
            with self.lock: self.state = dict(ready=False, error=self.error)

    def _run(self):
        path = Path(self.args.checkpoint)
        paused, speed, runtime = False, 10, 0.
        if path.exists():
            data = json.loads(path.read_text())
            game = GrandPrix.restore(data)
            settings = data.get('settings', {})
            paused, speed, runtime = settings.get('paused',False), settings.get('speed',10), settings.get('runtime',0.)
            self.config = public_config(game.candidates)
        else:
            game = GrandPrix(candidates=self.args.candidates, seed=self.args.seed)
        trial = None
        trial_path = path.with_name('grand-prix-holdout.json')
        if game.winner and trial_path.exists():
            try: trial = HoldoutTrial.restore(game, json.loads(trial_path.read_text()))
            except (ValueError, KeyError, TypeError): pass
        last_holdout = trial.summary() if trial else None
        measured, rate = 0, 0.
        last = clock = saved = runtime_clock = time.monotonic()
        dirty = True
        while not self.stop.is_set():
            tick = time.monotonic()
            if not paused and not game.winner:
                runtime += tick-runtime_clock
            runtime_clock = tick
            while True:
                try: command, done, result = self.commands.get_nowait()
                except queue.Empty: break
                try:
                    action = command.get('action')
                    if action=='pause': paused=bool(command['value'])
                    elif action=='speed':
                        if command.get('value') not in (1,3,10,30): raise ValueError('无效倍率')
                        speed=command['value']
                    elif action=='mutation':
                        value=float(command['value'])
                        if not .02<=value<=.5: raise ValueError('无效变异概率')
                        game.mutation=value
                    elif action=='save':
                        self.checkpoint(game,path,paused=paused,speed=speed,runtime=runtime)
                        with self.lock: self.model=json.loads(path.read_text())
                    elif action=='holdout':
                        trial=HoldoutTrial(game, command.get('value'))
                        paused=False
                        trial.save(trial_path)
                    elif action=='close_holdout':
                        if trial: trial.save(trial_path)
                        last_holdout=trial.summary() if trial else last_holdout
                        trial=None
                    elif action=='reset':
                        self.checkpoint(game,path,paused=paused,speed=speed,runtime=runtime)
                        path.replace(path.with_name(f'grand-prix-archive-{time.time_ns()}.json'))
                        game=GrandPrix(candidates=game.candidates,seed=time.time_ns()%(2**31))
                        trial=None
                        last_holdout=None
                        paused,runtime=False,0.
                    else: raise ValueError('无效赛事操作')
                    dirty=True
                    result['ok']=True
                except (ValueError,TypeError,KeyError) as exc: result['error']=str(exc)
                finally: done.set()
            if not paused and not game.winner:
                for _ in range(speed):
                    dirty = game.step() or dirty
                    measured+=1
                    if game.winner: break
            if trial is not None and not paused and not trial.arena.done():
                for _ in range(speed):
                    finished=trial.step()
                    measured+=1
                    if finished:
                        trial.save(trial_path)
                        dirty=True
                        break
            now=time.monotonic()
            if now-clock>=1:
                rate=measured/(now-clock); measured,clock=0,now
            if now-last>=.08 or dirty:
                state=trial.snapshot(game) if trial is not None else game.snapshot()
                state['lastHoldout']=trial.summary() if trial else last_holdout
                state.update(paused=paused, speed=speed, device=self.device_info,
                             stepsPerSecond=round(rate,1), gpuMemoryMB=round(torch.cuda.memory_allocated()/1048576,1), runtime=round(runtime,1))
                with self.lock: self.state=state
                last=now
            if dirty or now-saved>=30:
                self.checkpoint(game,path,paused=paused,speed=speed,runtime=runtime)
                if trial: trial.save(trial_path)
                # Exports are immutable CPU data refreshed at checkpoints, never CUDA objects.
                with self.lock: self.model=json.loads(path.read_text())
                saved=now; dirty=False
            self.stop.wait(max(0,1/30-(time.monotonic()-tick)))
        self.checkpoint(game,path,paused=paused,speed=speed,runtime=runtime)
        if trial: trial.save(trial_path)
