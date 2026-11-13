import copy
import unittest
import numpy as np
from f1_config import circuit, public_config
try:
 import torch
except ImportError:
 torch=None

class CircuitTests(unittest.TestCase):
 def test_adjacent_roads_keep_an_infield_gap(self):
  from f1_config import WIDTH_METERS, CELL
  for track in range(5):
   points,mask,_,_=circuit(track)
   distance=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
   indices=np.arange(512)
   apart=np.abs(indices[:,None]-indices[None,:]);apart=np.minimum(apart,512-apart)
   spacing=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1).mean()
   distance[apart*spacing<=90]=np.inf
   i,j=np.unravel_index(distance.argmin(),distance.shape)
   self.assertGreater(float(distance[i,j]),(WIDTH_METERS[track]+1.4)*circuit(track)[2])
   if track in (0,3):
    midpoint=(points[i]+points[j])/2
    self.assertFalse(mask[int(midpoint[1]/CELL),int(midpoint[0]/CELL)])
 def test_real_five_circuits_and_current_grid(self):
  c=public_config()
  self.assertEqual(len(c['drivers']),22);self.assertEqual(len(c['teams']),11)
  self.assertEqual(c['tracks'][0]['id'],'shanghai')
  self.assertEqual(len({t['id'] for t in c['tracks']}),6)
  self.assertEqual(sum(t['role']=='training' for t in c['tracks']),5)
  self.assertEqual(c['tracks'][5]['role'],'holdout')
  self.assertIn('CAM',[d['code'] for d in c['drivers']])
  for i in range(5):
   p,m,s,l=circuit(i)
   from f1_config import CELL
   world_length=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum()
   self.assertAlmostEqual(float(world_length/s),l,delta=.01)
   self.assertEqual(p.shape,(512,2)); self.assertTrue(np.isfinite(p).all())
   self.assertTrue(m[int(p[0,1]/CELL),int(p[0,0]/CELL)])
   self.assertGreater(l,5000)

@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'Requires actual CUDA')
class GrandPrixTests(unittest.TestCase):
 def game(self):
  from grand_prix import GrandPrix
  return GrandPrix(candidates=8)
 def test_metre_scale_and_full_body_collision(self):
  from f1_config import CAR_LENGTH,CAR_WIDTH,WIDTH_METERS
  from grand_prix import Arena
  g=self.game()
  for track in range(5):
   arena=Arena(g.champions.clone(),track,3,g.assets)
   self.assertGreater(WIDTH_METERS[track]/CAR_WIDTH,5)
   corners=arena.body_corners
   self.assertAlmostEqual((corners[:,0].max()-corners[:,0].min()).item(),CAR_LENGTH,places=5)
   self.assertAlmostEqual((corners[:,1].max()-corners[:,1].min()).item(),CAR_WIDTH,places=5)
   arena.step()
   self.assertTrue(torch.isfinite(arena.xy).all().item())
   self.assertTrue(arena.inside(arena.xy).all().item())
 def test_cuda_and_exact_checkpoint_continuation(self):
  from grand_prix import GrandPrix
  g=self.game()
  for _ in range(16):g.step()
  self.assertEqual(g.arena.xy.device.type,'cuda')
  s=g.snapshot(); self.assertEqual(len(s['cars']),22); self.assertEqual(s['networks'],176)
  restored=GrandPrix.restore(g.save())
  for _ in range(12):g.step();restored.step()
  self.assertTrue(torch.equal(g.arena.xy,restored.arena.xy))
  self.assertTrue(torch.equal(g.weights,restored.weights))
  bad=g.save();bad['champions'][0][0]=5
  with self.assertRaises(ValueError):GrandPrix.restore(bad)
 def test_no_cross_driver_genes(self):
  g=self.game();g.mutation=0
  for i in range(22):g.weights[i].fill_(i/22)
  g.champions=g.weights[:,0].clone();g.evolve()
  for i in range(22):self.assertTrue(torch.all(g.weights[i,:-1]==i/22).item())
 def test_geometry_migration_preserves_models_and_rechecks_certificates(self):
  from grand_prix import GrandPrix
  g=self.game();g.results.fill_(3);g.pass_ticks.fill_(100)
  data=g.save();data['geometryVersion']=1;data['winner']={'drivers':[0]}
  restored=GrandPrix.restore(data)
  self.assertTrue(torch.equal(restored.champions,g.champions))
  self.assertTrue(torch.equal(restored.weights[:,0],g.champions))
  self.assertFalse((restored.pass_ticks>=0).any().item())
  self.assertIsNone(restored.winner)
  self.assertEqual((restored.phase,restored.stage),('training',0))
 def test_live_representative_stays_until_its_attempt_ends(self):
  g=self.game();first=g.snapshot()['cars'][0]['candidate']
  g.arena.best[1]=100
  self.assertEqual(g.snapshot()['cars'][0]['candidate'],first)
  g.arena.reason[first]=1
  self.assertEqual(g.snapshot()['cars'][0]['candidate'],1)
 def test_validation_does_not_mutate_and_requires_all_five(self):
  from grand_prix import Arena
  g=self.game();g.phase='validation';g.stage=4
  g.arena=Arena(g.champions.clone(),4,3,g.assets)
  before=g.champions.clone();g.arena.reason[0]=4;g.arena.finish_tick[0]=100;g.arena.best[0]=1536
  g.update_validation();self.assertIsNone(g.winner)
  g.pass_ticks[0,:4]=200;g.total_steps=1000;g.arena.ticks=104
  g.update_validation();self.assertEqual(g.winner['drivers'],[0])
  self.assertEqual(g.winner['certificates'][0]['modelHash'],g.hashes()[0])
  self.assertEqual(g.winner['certificates'][0]['laps'],[3]*5)
  self.assertTrue(torch.equal(g.champions,before))
 def test_same_tick_tie_break_and_earliest_finish(self):
  from grand_prix import Arena
  g=self.game();g.phase='validation';g.stage=4;g.arena=Arena(g.champions.clone(),4,3,g.assets)
  g.pass_ticks[:3,:4]=200;g.arena.reason[:3]=4;g.arena.finish_tick[:3]=torch.tensor([104,102,102],device='cuda')
  g.pass_ticks[2,0]=190;g.total_steps=1000;g.arena.ticks=104;g.update_validation()
  self.assertEqual(g.winner['drivers'],[2])
 def test_body_outside_road_is_rejected_even_with_center_inside(self):
  from grand_prix import Arena
  from f1_config import WIDTH_METERS,CAR_WIDTH
  g=self.game();arena=Arena(torch.zeros_like(g.champions),0,3,g.assets)
  tangent=arena.points[1]-arena.points[0];tangent=tangent/torch.linalg.vector_norm(tangent)
  normal=torch.stack((-tangent[1],tangent[0]))
  offset=(WIDTH_METERS[0]/2-CAR_WIDTH/4)*arena.scale
  arena.xy[:]=arena.points[0]+normal*offset
  self.assertLess(offset,WIDTH_METERS[0]/2*arena.scale)
  arena.step()
  self.assertTrue((arena.reason==1).all().item())
 def test_legacy_inflight_validation_waits_another_five_rounds(self):
  from grand_prix import GrandPrix
  g=self.game();g.generation=14;g.phase='validation'
  from grand_prix import Arena
  g.arena=Arena(g.champions.clone(),0,3,g.assets)
  data=g.save();data.pop('nextValidationGeneration')
  g=GrandPrix.restore(data);g.evolve()
  self.assertEqual(g.next_validation_generation,19)
  self.assertEqual(g.generation,15)
  for generation in range(15,19):
   g.stage=4;g.finish_training_stage()
   self.assertEqual(g.generation,generation+1)
   self.assertEqual(g.phase,'training')
  g.stage=4;g.finish_training_stage()
  self.assertEqual((g.generation,g.phase),(19,'validation'))
 def test_validation_only_after_each_five_training_rounds(self):
  from grand_prix import GrandPrix
  g=self.game()
  for expected in range(1,5):
   self.assertEqual(g.generation,expected)
   g.stage=4;g.results.fill_(3);g.pass_ticks.fill_(100)
   g.finish_training_stage()
   self.assertEqual(g.phase,'training');self.assertEqual(g.stage,0)
   self.assertEqual(g.generation,expected+1)
   self.assertFalse((g.pass_ticks>=0).any().item())
  g.stage=4;g.finish_training_stage()
  self.assertEqual((g.phase,g.generation),('validation',5))
  restored=GrandPrix.restore(g.save())
  self.assertEqual((restored.phase,restored.generation),('validation',5))
  restored.evolve()
  for expected in range(6,10):
   restored.stage=4;restored.finish_training_stage()
   self.assertEqual((restored.phase,restored.generation),('training',expected+1))
  restored.stage=4;restored.finish_training_stage()
  self.assertEqual((restored.phase,restored.generation),('validation',10))
 def test_new_model_resets_validation_and_trial_is_bounded(self):
  g=self.game();g.generation=5;g.stage=4;g.results.fill_(3);g.pass_ticks.fill_(100)
  g.finish_training_stage()
  self.assertEqual(g.phase,'validation');self.assertFalse((g.pass_ticks>=0).any().item())
  g.arena.ticks=100000;g.arena.step()
  self.assertEqual(g.arena.xy.shape,(22,2))
  self.assertTrue(torch.isfinite(g.arena.xy).all().item())

if __name__=='__main__':unittest.main()

@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'Requires actual CUDA')
class HoldoutTests(unittest.TestCase):
 def fixture(self):
  from grand_prix import GrandPrix
  g=GrandPrix(candidates=8)
  g.winner={'drivers':[0], 'certificates':[{'driver':0,'modelHash':g.model_hashes[0],'weights':g.champions[0].cpu().tolist()}]}
  return g
 def test_unseen_trial_preserves_entire_championship_and_weights(self):
  import json
  from grand_prix import HoldoutTrial
  g=self.fixture()
  self.assertEqual(len(g.assets),5)
  self.assertEqual(tuple(g.scores.shape),(22,8,5))
  before=json.dumps(g.save(),sort_keys=True)
  t=HoldoutTrial(g)
  frozen=t.arena.weights.clone()
  for _ in range(30):t.step()
  s=t.snapshot(g)
  self.assertEqual(s['track'],5)
  self.assertEqual(sum(not c['hiddenCar'] for c in s['cars']),1)
  self.assertTrue(torch.equal(frozen,t.arena.weights))
  self.assertTrue(t.summary()['weightsUnchanged'])
  self.assertEqual(json.dumps(g.save(),sort_keys=True),before)
 def test_failed_trial_and_checkpoint_roundtrip(self):
  import tempfile,json
  from pathlib import Path
  from grand_prix import HoldoutTrial
  g=self.fixture();t=HoldoutTrial(g)
  t.arena.xy.fill_(-1000);t.step()
  self.assertEqual(t.summary()['status'],'failed')
  self.assertEqual(t.summary()['reason'],'驶出有效路面')
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'holdout.json';t.save(path)
   restored=HoldoutTrial.restore(g,json.loads(path.read_text()))
   self.assertEqual(restored.summary(),t.summary())
   before=restored.arena.ticks;restored.step()
   self.assertEqual(restored.arena.ticks,before)
 def test_three_laps_required_and_invalid_certificate_rejected(self):
  from grand_prix import HoldoutTrial
  g=self.fixture();t=HoldoutTrial(g)
  self.assertEqual(t.arena.laps,3)
  t.arena.reason[0]=4;t.arena.best[0]=1536;t.arena.lap_ticks[0]=torch.tensor([100,230,360],device='cuda')
  self.assertEqual(t.summary()['status'],'passed')
  self.assertEqual(t.summary()['lapSeconds'],[3.33,4.33,4.33])
  g.winner['certificates'][0]['weights'][0]+=.1
  with self.assertRaises(ValueError):HoldoutTrial(g)
  g.winner=None
  with self.assertRaises(ValueError):HoldoutTrial(g)
