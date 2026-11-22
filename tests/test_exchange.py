"""校验资料入口能加载业务样本并拒绝缺少关键字段的文件。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_exchange import inspect


class ExchangeTests(unittest.TestCase):
    def test_scene_is_readable(self):
        path = Path(__file__).resolve().parents[1] / "samples" / "scene.json"
        result = inspect(path)
        self.assertIn('revision', result["字段"])
        self.assertEqual(result["数值环境"], "2.2.6")

    def test_alternate_exchange_is_readable(self):
        path = Path(__file__).resolve().parents[1] / "samples" / "axis-revision.json"
        result = inspect(path)
        self.assertGreater(result["字节"], 0)

    def test_incomplete_scene_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            with self.assertRaises(KeyError):
                inspect(path)
