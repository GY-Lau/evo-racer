import json
import math
import unittest
import numpy as np
from f1_config import circuit, WIDTH, HEIGHT, POINTS
try:
 import torch
except ImportError:
 torch=None

@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'Requires actual CUDA')
class PhysicsExecutionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.assets=[]
  for i in range(5):
   p,m,s,l=circuit(i);cls.assets.append((torch.as_tensor(p,device='cuda'),torch.as_tensor(m,device='cuda'),s,l))
 def test_graph_matches_eager_with_compaction_and_freezes_retired_models(self):
  from grand_prix import Arena
  weights=torch.rand((176,112),generator=torch.Generator(device='cuda').manual_seed(74),device='cuda')*2-1
  eager=Arena(weights.clone(),0,3,self.assets,optimized=False)
  graph=Arena(weights.clone(),0,3,self.assets)
  # Keep a stable runner and force other cars out to exercise compaction.
  for a in (eager,graph):a.xy[1:]=torch.tensor([-500.,-500.],device='cuda')
  for _ in range(160):eager.step();graph.step()
  self.assertLess(graph.active_count,176)
  for name in Arena.STATE_NAMES:
   self.assertTrue(torch.equal(getattr(eager,name),getattr(graph,name)),name)
  self.assertEqual(eager.ticks,graph.ticks)
 def test_graph_exact_checkpoint_continuation_and_warmup_does_not_advance(self):
  from grand_prix import Arena
  weights=torch.zeros((22,112),device='cuda')
  a=Arena(weights,0,3,self.assets)
  for _ in range(20):a.step()
  b=Arena(weights.clone(),0,3,self.assets);b.restore(a.save())
  for _ in range(40):a.step();b.step()
  self.assertEqual(a.ticks,60);self.assertEqual(b.tick_tensor.item(),60)
  for name in Arena.STATE_NAMES:self.assertTrue(torch.equal(getattr(a,name),getattr(b,name)),name)
 def test_finite_acceleration_braking_grip_circle_and_steering_response(self):
  from grand_prix import Arena,DT,ENGINE_ACCEL,BASE_GRIP,DOWNFORCE_GRIP,GRAVITY,STEER_RATE
  speed=torch.tensor([85.,85.,25.],device='cuda')
  zero=torch.zeros_like(speed)
  outputs=torch.tensor([[1.,1.],[1.,-1.],[1.,1.]],device='cuda')
  v,yaw,steer,ax,ay,usage,throttle,brake=Arena.dynamics(speed,zero,zero,outputs)
  self.assertLessEqual(float((v-speed).max()),ENGINE_ACCEL*DT+1e-5)
  self.assertLess(float(v[1]),85.)
  self.assertGreater(float(brake[1]),.9)
  self.assertTrue((steer.abs()<=STEER_RATE*DT+1e-6).all().item())
  for _ in range(60):v,yaw,steer,ax,ay,usage,throttle,brake=Arena.dynamics(speed,yaw,steer,outputs)
  self.assertTrue((usage<=1.00001).all().item())
  self.assertLess(float(v[0]/yaw[0]),float(v[2]/yaw[2])*10) # Both radii are finite.
  self.assertGreater(float(v[0]/yaw[0]),float(v[2]/yaw[2]))
 def test_fast_hard_corner_understeers_while_slow_runner_completes_circle(self):
  from grand_prix import Arena,WHEELBASE,STEER_LIMIT,CAR_WIDTH,WIDTH_METERS
  radius=70.;scale=2.
  theta=np.arange(POINTS)*2*np.pi/POINTS
  p=np.stack((700+radius*scale*np.cos(theta),420+radius*scale*np.sin(theta)),axis=1).astype(np.float32)
  yy,xx=np.mgrid[:HEIGHT,:WIDTH]
  mask=np.abs(np.sqrt((xx+.5-700)**2+(yy+.5-420)**2)-radius*scale)<WIDTH_METERS[0]*scale/2
  assets=[(torch.as_tensor(p,device='cuda'),torch.as_tensor(mask,device='cuda'),scale,2*np.pi*radius)]
  weights=torch.zeros((2,112),device='cuda')
  weights[:,100]=math.atanh(math.atan(WHEELBASE/radius)/STEER_LIMIT)
  for i,speed in enumerate((80.,25.)):
   weights[i,111]=math.atanh((speed-8)/43.5-1)
  a=Arena(weights,0,1,assets);a.speed[:]=torch.tensor([80.,25.],device='cuda')
  for _ in range(750):
   a.step()
   if a.ticks%4==0 and a.done():break
  self.assertEqual(int(a.reason[0]),1,'Fast car must understeer off the road')
  self.assertEqual(int(a.reason[1]),4,'Slower car must complete the same bend')
  self.assertLess(float(a.best[0]),POINTS)
 def test_validation_skips_ineligible_drivers_and_ends_when_all_fail(self):
  from grand_prix import GrandPrix,Arena
  g=GrandPrix(candidates=8);g.phase='validation';g.generation=5;g.stage=0
  g.arena=Arena(g.champions.clone(),0,3,g.assets)
  g.arena.reason[:]=1;g.arena.reason[0]=4;g.arena.best[0]=1536;g.arena.finish_tick[0]=10
  g.arena.ticks=3;g.arena.tick_tensor.fill_(3)
  g.step()
  self.assertEqual(g.stage,1)
  self.assertEqual(int(g.arena.reason[0]),0)
  self.assertTrue((g.arena.reason[1:]==6).all().item())
  self.assertTrue(g.validation_skipped[1:,1:].all())
  g.arena.reason[0]=1;g.arena.ticks=3;g.arena.tick_tensor.fill_(3)
  g.step()
  self.assertEqual((g.phase,g.generation,g.stage),('training',6,0))
  self.assertEqual(g.next_validation_generation,10)
 def test_physics_migration_invalidates_old_proof_preserves_weights(self):
  from grand_prix import GrandPrix,PHYSICS_VERSION
  g=GrandPrix(candidates=8);data=g.save();data['physicsVersion']=1;data['winner']={'drivers':[0]}
  original=g.champions.clone()
  restored=GrandPrix.restore(data)
  self.assertIsNone(restored.winner)
  self.assertTrue(torch.equal(restored.champions,original))
  self.assertEqual(restored.save()['physicsVersion'],PHYSICS_VERSION)
