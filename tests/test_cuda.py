import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "需要真实 CUDA GPU 和 CUDA 版 PyTorch")
class CudaTests(unittest.TestCase):
    def test_network_and_simulation_really_use_cuda(self):
        from cuda_training import CudaEvolution
        sim = CudaEvolution(seed=5)
        for _ in range(20): sim.step()
        torch.cuda.synchronize()
        for value in (sim.weights, sim.xy, sim.rays, sim.hidden, sim.outputs, sim.best):
            self.assertEqual(value.device.type, 'cuda')
            self.assertTrue(torch.isfinite(value).all().item())
        weights = sim.weights[0].cpu().numpy()
        inputs = sim.inputs[0].cpu().numpy()
        first = weights[:90].reshape(10, 9)
        hidden = np.tanh(first[:, :8] @ inputs + first[:, 8])
        second = weights[90:].reshape(2, 11)
        expected = np.tanh(second[:, :10] @ hidden + second[:, 10])
        np.testing.assert_allclose(sim.outputs[0].cpu().numpy(), expected, atol=1e-4)
        state = sim.snapshot()
        self.assertEqual(len(state['cars']), 20)
        leader = next(c for c in state['cars'] if c['id'] == state['leaderId'])
        self.assertEqual(len(leader['weights']), 112)

    def test_selection_preserves_elite_and_changes_children(self):
        from cuda_training import CudaEvolution
        sim = CudaEvolution(seed=7)
        sim.best[:] = torch.arange(20, device='cuda')
        champion, original = sim.weights[19].clone(), sim.weights.clone()
        sim.next_generation()
        self.assertEqual(sim.info.generation, 2)
        self.assertEqual(sim.weights.shape, (20, 112))
        self.assertTrue(torch.equal(sim.weights[0], champion))
        self.assertFalse(torch.equal(original, sim.weights))
        self.assertEqual(sim.history[0]['best'], 19)
        self.assertEqual(sim.xy.device.type, 'cuda')

    def test_model_roundtrip_and_no_partial_invalid_import(self):
        from cuda_training import CudaEvolution
        sim = CudaEvolution(seed=12, kind='technical')
        for _ in range(8): sim.step()
        model = sim.export_model()
        restored = CudaEvolution(seed=13)
        restored.import_model(model)
        self.assertEqual(restored.kind, 'technical')
        self.assertEqual(restored.info.record, model['record'])
        self.assertTrue(torch.equal(restored.weights[0], sim.champion))
        with self.assertRaises(ValueError): restored.import_model({})
        self.assertEqual(restored.kind, 'technical')

    def test_continuous_trial_exceeds_training_limits_with_bounded_memory(self):
        from cuda_training import CudaEvolution
        model = {'format': 'evo-racer-cuda-v1', 'track': 'flow', 'record': 841., 'weights': [0.] * 112}
        trial = CudaEvolution(population=1, trial=True)
        trial.import_model(model)
        weights = trial.weights.clone()
        # Cross both limits without needing a particular evolved driving policy.
        trial.ticks = 1200
        trial.best[0] = trial.progress[0] = 1260
        trial.step()
        self.assertGreater(trial.info.elapsed, 40)
        self.assertFalse(trial.finished())
        self.assertIsNone(trial.snapshot()['limit'])
        self.assertEqual(trial.trajectory.shape, (1, 1, 2))
        trial.ticks = 2000000
        trial.step()
        self.assertEqual(trial.trajectory.shape, (1, 1, 2))
        self.assertTrue(torch.equal(trial.weights, weights))
        self.assertEqual(trial.export_model(), model)
        with self.assertRaises(ValueError): trial.next_generation()

    def test_trial_stops_on_real_road_exit_and_reports_reason(self):
        from cuda_training import CudaEvolution
        trial = CudaEvolution(population=1, trial=True)
        trial.weights.zero_()  # A straight-driving policy eventually leaves the curved road.
        for _ in range(300):
            trial.step()
            if trial.finished(): break
        self.assertTrue(trial.finished())
        self.assertEqual(trial.end_reason(), '驶出有效路面')
        state = trial.snapshot()
        self.assertEqual(len(state['cars']), 1)
        self.assertFalse(state['cars'][0]['alive'])
        trial._spawn()
        trial.idle[0], trial.best[0] = 4.1, 100
        trial.step()
        self.assertEqual(trial.end_reason(), '连续 4 秒未刷新最远进度')
        trial._spawn()
        trial.progress[0] = -13
        trial.step()
        self.assertEqual(trial.end_reason(), '倒退越过起点')

    def test_trial_publish_keeps_training_record_population_and_model(self):
        from cuda_training import CudaEvolution
        from server import Trainer
        from types import SimpleNamespace
        sim = CudaEvolution(seed=12)
        for _ in range(8): sim.step()
        model = sim.export_model()
        original = sim.weights.clone()
        trial = CudaEvolution(population=1, trial=True)
        trial.import_model(model)
        trial.best[0] = trial.progress[0] = 1260
        trainer = Trainer(SimpleNamespace(), {'verified': True})
        trainer.publish(sim, trial=trial, paused=False, speed=1, steps_per_second=30)
        self.assertEqual(trainer.state['mode'], 'champion')
        self.assertEqual(trainer.state['trial']['laps'], 3)
        self.assertEqual(trainer.state['population'], 1)
        self.assertEqual(trainer.state['generation'], sim.info.generation)
        self.assertEqual(trainer.state['record'], model['record'])
        self.assertEqual(trainer.model, model)
        self.assertTrue(torch.equal(trial.weights[0], sim.champion))
        self.assertTrue(torch.equal(sim.weights, original))

    def test_training_evaluation_limits_still_apply(self):
        from cuda_training import CudaEvolution
        sim = CudaEvolution()
        sim.best[0] = 840
        self.assertTrue(sim.finished())
        reason = sim.end_reason()
        self.assertIn('两圈', reason)
        sim.next_generation(reason=reason)
        self.assertEqual(sim.snapshot()['lastEndReason'], reason)
        sim.ticks = 1200
        self.assertTrue(sim.finished())
        self.assertIn('40 秒', sim.end_reason())


if __name__ == '__main__':
    unittest.main()
