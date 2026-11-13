import copy
import json
import unittest
from email.message import Message
from types import SimpleNamespace
from threading import Lock

from race_common import geometry, validate_model, WEIGHTS
from server import Handler


class CommonTests(unittest.TestCase):
    def test_road_mask_start_and_boundaries(self):
        for kind in ("flow", "technical"):
            points, mask, width = geometry(kind)
            self.assertEqual(points.shape, (420, 2))
            self.assertEqual(mask.shape, (234, 367))
            self.assertTrue(mask[int(points[0, 1] // 3), int(points[0, 0] // 3)])
            self.assertFalse(mask[0, 0])
            self.assertFalse(mask[116, 183])
            self.assertLess(width, 100)

    def test_valid_model_roundtrip(self):
        data = {"format": "evo-racer-cuda-v1", "weights": [0.] * WEIGHTS, "track": "flow", "record": 31.}
        self.assertEqual(validate_model(json.loads(json.dumps(data))), data)

    def test_invalid_models_rejected(self):
        good = {"format": "evo-racer-cuda-v1", "weights": [0.] * WEIGHTS, "track": "flow", "record": 31.}
        bad = [None, [], {}, {**good, "weights": [0.]}, {**good, "track": "elsewhere"}, {**good, "record": float("nan")}, {**good, "format": "evo-racer-v1"}]
        for value in (float("inf"), float("nan"), True, 4.1, "0"):
            model = copy.deepcopy(good)
            model["weights"][8] = value
            bad.append(model)
        for data in bad:
            with self.subTest(data=str(data)[:60]), self.assertRaises(ValueError):
                validate_model(data)

    def test_http_snapshot_contract_without_network(self):
        handler = Handler.__new__(Handler)
        handler.path = "/api/state"
        handler.server = SimpleNamespace(trainer=SimpleNamespace(lock=Lock(), state={"ready": True, "device": {"verified": True}}))
        replies = []
        handler.send_json = lambda data, status=200: replies.append((status, data))
        handler.do_GET()
        self.assertEqual(replies[0][0], 200)
        self.assertTrue(replies[0][1]["device"]["verified"])

    def test_cross_origin_control_rejected(self):
        handler = Handler.__new__(Handler)
        handler.path = "/api/control"
        handler.headers = Message()
        handler.headers['Origin'] = 'http://other-site.example'
        handler.headers['Host'] = '192.168.110.186:8765'
        replies = []
        handler.send_json = lambda data, status=200: replies.append((status, data))
        handler.do_POST()
        self.assertEqual(replies[0][0], 403)


if __name__ == "__main__":
    unittest.main()

class ArchiveHttpTests(unittest.TestCase):
    def test_archive_list_download_and_path_restriction(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=root/'123';record.mkdir()
            manifest=dict(id='123',seed=7,winner={'drivers':[12]},runtime=5510.7,holdout={'status':'passed'})
            (record/'manifest.json').write_text(json.dumps(manifest))
            model={'original':'unchanged','weights':[.25]}
            (record/'grand-prix.json').write_text(json.dumps(model))
            handler=Handler.__new__(Handler)
            handler.server=SimpleNamespace(trainer=SimpleNamespace(records_dir=root))
            replies=[];handler.send_json=lambda data,status=200:replies.append((status,data))
            for url in ('/api/records','/api/records/123/model','/api/records/../model','/api/records/123/manifest','/api/records/123/no/file'):
                handler.path=url;handler.do_GET()
            self.assertEqual(replies[0],(200,[manifest]))
            self.assertEqual(replies[1],(200,model))
            self.assertEqual([r[0] for r in replies[2:]],[404,404,404])
            self.assertEqual(json.loads((record/'grand-prix.json').read_text()),model)
