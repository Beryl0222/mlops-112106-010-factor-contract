"""执行器测试套件。

覆盖交换约定中的硬性语义：命名轴对齐（同形不换轴）、结构零跳过、
缺块不冒充零、压缩/原值双视图与误差契约、报警原子交付、趋势局部交付、
工作区额度、共享子表达式的实际节省与缓存失效，以及跨修订不变性。
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensor_exec.contraction import (  # noqa: E402
    apply_private,
    build_shared,
    execute_head,
)
from tensor_exec.engine import ExecutionEngine  # noqa: E402
from tensor_exec.intervals import term_product  # noqa: E402
from tensor_exec.package import ModelPackage, Slice  # noqa: E402
from reference import reference  # noqa: E402

SCENE = ROOT / "samples" / "scene.json"
AXIS_REV = ROOT / "samples" / "axis-revision.json"


def load(path: Path) -> ModelPackage:
    return ModelPackage(json.loads(path.read_text(encoding="utf-8")))


def base_raw() -> dict:
    return json.loads(SCENE.read_text(encoding="utf-8"))


class ModelAndBlockTests(unittest.TestCase):
    def test_named_axes_not_swapped_when_shapes_match(self):
        """两个同形因子只能按逻辑轴名对齐，物理轴序交换后数值不变。"""
        p7 = load(SCENE)
        p8 = load(AXIS_REV)
        # p8 的 frequency 物理轴序与 p7 相反。
        self.assertEqual(
            p7.factors["frequency"].physical_axes, ("r1", "frequency"))
        self.assertEqual(
            p8.factors["frequency"].physical_axes, ("frequency", "r1"))
        for hn in p7.heads:
            for i in range(len(p7.heads[hn].slices)):
                r7 = execute_head(p7.heads[hn], p7,
                                  p7.heads[hn].slices[i], i, False)
                r8 = execute_head(p8.heads[hn], p8,
                                  p8.heads[hn].slices[i], i, False)
                np.testing.assert_allclose(r7.center, r8.center, atol=1e-12)
                np.testing.assert_allclose(r7.bound, r8.bound, atol=1e-12)

    def test_structural_zero_distinguished_from_missing(self):
        raw = base_raw()
        pkg = ModelPackage(raw)
        zero_block = next(
            b for b in pkg.factors["input"].blocks if b.zero)
        self.assertTrue(zero_block.zero)
        # 人为制造一个缺块：删除一个有数据块后网格应出现 None。
        raw2 = copy.deepcopy(raw)
        del raw2["factors"]["channel"]["blocks"][0]
        pkg2 = ModelPackage(raw2)
        missing = [b for _r, b in pkg2.grid_cells("channel") if b is None]
        self.assertTrue(missing)

    def test_overlapping_blocks_rejected(self):
        raw = base_raw()
        raw["factors"]["input"]["blocks"].append(copy.deepcopy(
            raw["factors"]["input"]["blocks"][0]))
        with self.assertRaises(ValueError):
            ModelPackage(raw)

    def test_compressed_view_decode_and_error(self):
        pkg = load(SCENE)
        block = pkg.factors["input"].blocks[0]
        values, error, name = block.view(prefer_original=False)
        np.testing.assert_allclose(
            values, np.asarray(block.center) * block.scale)
        self.assertEqual(error, block.error)
        self.assertEqual(name, "compressed")

    def test_term_product_includes_second_order_terms(self):
        c = [np.array([1.0]), np.array([2.0])]
        e = [np.array([0.1]), np.array([0.2])]
        _, abs_prod, worst = term_product(c, e)
        # (1+0.1)(2+0.2) - 1*2 = 2.42 - 2 = 0.42（含交叉项 0.1*0.2）
        np.testing.assert_allclose(worst - abs_prod, [0.42], atol=1e-15)


class ContractionCorrectnessTests(unittest.TestCase):
    def test_matches_brute_force_reference_all_heads(self):
        for path in (SCENE, AXIS_REV):
            pkg = load(path)
            for hn in pkg.heads:
                head = pkg.heads[hn]
                value, bound, _ = reference(pkg, hn, False)
                for i, sl in enumerate(head.slices):
                    result = execute_head(head, pkg, sl, i, False)
                    idx = tuple(slice(lo, hi) for lo, hi
                                in sl.region(head.axes, pkg.lengths))
                    np.testing.assert_allclose(
                        result.center, value[idx], atol=1e-12)
                    # 执行器界 = 输入误差界 + 非负舍入，必须覆盖参考输入界。
                    self.assertTrue(
                        np.all(result.bound + 1e-12 >= bound[idx]))

    def test_structural_zero_skips_macs(self):
        pkg = load(SCENE)
        head = pkg.heads["alarm"]
        result = execute_head(head, pkg, head.slices[0], 0, False)
        # r0 的结构零行不产生乘加：只有 r0∈[0,2) × r1∈[0,2)。
        self.assertGreater(result.macs, 0)
        # 把结构零块换成“数值为零的普通块”时乘加应增加（零值仍需计算）。
        raw = base_raw()
        for block in raw["factors"]["input"]["blocks"]:
            if block.get("zero"):
                block.pop("zero")
                block["center"] = [[0, 0]]
                block["scale"] = 0.1
                block["error"] = 0.0
        pkg2 = ModelPackage(raw)
        result2 = execute_head(pkg2.heads["alarm"], pkg2,
                               pkg2.heads["alarm"].slices[0], 0, False)
        self.assertGreater(result2.macs, result.macs)

    def test_missing_block_makes_output_unprovable(self):
        raw = base_raw()
        # 清空 channel 因子的全部块 -> 缺块，不能当零。
        raw["factors"]["channel"]["blocks"] = []
        pkg = ModelPackage(raw)
        head = pkg.heads["alarm"]
        result = execute_head(head, pkg, head.slices[0], 0, False)
        self.assertFalse(result.provable)
        self.assertTrue(result.missing)
        self.assertIsNone(result.center)

    def test_small_output_from_cancellation_still_judged_by_bound(self):
        """符号相消得到很小的输出，也不能仅凭值小判定合格。"""
        pkg = load(SCENE)
        # trend 切片1 中心值很小但压缩误差放大，界远超契约，应被拒。
        head = pkg.heads["trend"]
        result = execute_head(head, pkg, head.slices[1], 1, False)
        self.assertTrue(np.all(np.abs(result.center) < 0.1))
        self.assertGreater(result.max_error(), head.error_limit)
        self.assertFalse(result.within_limit(head.error_limit))

    def test_shared_equals_separate_numerically(self):
        pkg = load(SCENE)
        heads = [pkg.heads["alarm"], pkg.heads["trend"]]
        shared, missing = build_shared(heads, pkg, False)
        self.assertEqual(missing, [])
        # r1 被私有因子使用，必须保留；只有 r0 能提前消去。
        self.assertIn("r1", shared.axes)
        self.assertNotIn("r0", shared.axes)
        for head in heads:
            for i, sl in enumerate(head.slices):
                separate = execute_head(head, pkg, sl, i, False)
                via_shared = apply_private(
                    head, pkg, sl, i, shared, prefer_original=False)
                np.testing.assert_allclose(
                    separate.center, via_shared.center, atol=1e-12)
                np.testing.assert_allclose(
                    separate.bound, via_shared.bound, atol=1e-9)


class DeliverySemanticsTests(unittest.TestCase):
    def _engine(self, raw=None, budget=None):
        raw = raw or base_raw()
        return ExecutionEngine(
            ModelPackage(raw),
            budget if budget is not None else raw["budget_bytes"])

    def test_alarm_delivered_atomically_never_partial(self):
        engine = self._engine()
        engine.run_timeline(base_raw()["timeline"])
        # compressed_only 阶段 alarm 两个切片必须同时挂起。
        pending_alarm = [d for d in engine.ledger
                         if d.head == "alarm" and d.status == "pending"]
        delivered_early = [d for d in engine.ledger
                           if d.head == "alarm" and d.status == "delivered"
                           and d.event < 4]
        self.assertEqual(len(pending_alarm), 2)
        self.assertEqual(delivered_early, [])

    def test_alarm_delivers_together_after_originals_arrive(self):
        engine = self._engine()
        engine.run_timeline(base_raw()["timeline"])
        final = {idx: d for (h, idx), d in engine.delivered.items()
                 if h == "alarm"}
        self.assertEqual(set(final), {0, 1})
        for d in final.values():
            self.assertLessEqual(d.max_error, d.error_limit)
            self.assertEqual(d.revision, "p7")

    def test_trend_delivers_independent_slices(self):
        engine = self._engine()
        engine.run_timeline(base_raw()["timeline"])
        delivered_trend = {d.slice_index for d in engine.delivered.values()
                           if d.head == "trend"}
        self.assertIn(0, delivered_trend)  # 合格切片先交付
        rejected = [d for d in engine.ledger
                    if d.head == "trend" and d.status == "rejected"]
        self.assertTrue(rejected)  # 不合格切片被拒而非伪装成功

    def test_rejected_trend_slice_recoverable_after_original(self):
        raw = base_raw()
        # 让 frequency 原值也送达，收紧 trend 切片1 误差。
        timeline = raw["timeline"] + [
            ["arrive_original", "frequency"], ["request", "trend"]]
        engine = self._engine()
        engine.run_timeline(timeline)
        statuses = [d.status for d in engine.ledger
                    if d.head == "trend" and d.slice_index == 1]
        self.assertIn("delivered", statuses)

    def test_private_factor_uses_original_independently_of_shared_view(self):
        """共享中间量仍是压缩时，已补齐原值的私有因子必须独立采用原值。"""
        raw = json.loads(AXIS_REV.read_text(encoding="utf-8"))
        engine = ExecutionEngine(ModelPackage(raw), raw["budget_bytes"])
        for event in raw["timeline"]:
            engine.advance(event)
        second = [d for d in engine.ledger
                  if d.head == "trend" and d.slice_index == 1
                  and d.event == 4][0]
        self.assertIn("original", second.views_used["frequency"])
        # 误差界应较首次（全压缩）明显收紧。
        first = [d for d in engine.ledger
                 if d.head == "trend" and d.slice_index == 1
                 and d.event == 1][0]
        self.assertLess(second.max_error, first.max_error)

    def test_ledger_records_revision_and_reasons(self):
        engine = self._engine()
        engine.run_timeline(base_raw()["timeline"])
        report = engine.report()
        for group in ("已交付", "挂起", "拒绝"):
            for entry in report[group]:
                self.assertIn("模型包修订", entry)
                self.assertTrue(entry["原因"])
                self.assertIn("实际误差上界", entry)


class WorkspaceAndSharingTests(unittest.TestCase):
    def test_shared_plan_reduces_macs_or_reads(self):
        raw = base_raw()
        engine = ExecutionEngine(ModelPackage(raw), raw["budget_bytes"])
        engine.advance(["compressed_only"])
        engine.advance(["request", "trend"])
        report = engine.report()
        shared_decisions = [d for d in report["决策缓存"]
                            if d["方案"] == "shared"]
        self.assertTrue(shared_decisions)

    def test_shared_stage_executed_once_across_heads(self):
        """两个活动头消费同一公共子表达式时，共享乘加只实际执行一次。"""
        raw = base_raw()
        engine = ExecutionEngine(ModelPackage(raw), raw["budget_bytes"])
        # 同一视图级别（compressed_only 冻结期）下两个头先后请求，
        # 共享指纹相同，第二次必须命中缓存而非重建。
        engine.advance(["compressed_only"])
        engine.advance(["request", "trend"])
        engine.advance(["request", "alarm"])
        shared_entries = [d for d in engine.ledger
                          if d.used_shared and d.status in
                          ("delivered", "pending", "rejected")]
        nonzero_stage = [d for d in shared_entries
                         if d.shared_stage_macs > 0]
        # 同一指纹的共享阶段在冻结期内只被计费一次（仅首个消费头）。
        self.assertEqual({d.head for d in nonzero_stage}, {"trend"})
        self.assertEqual(len(engine._shared_cache), 1)
        # 之后的消费方（alarm 两个切片）命中缓存，共享阶段乘加为 0，
        # 只做各自私有部分——公共收缩没有重复执行。
        cached = [d for d in shared_entries
                  if d.head == "alarm" and d.shared_stage_macs == 0
                  and d.macs > 0]
        self.assertEqual(len(cached), 2)

    def test_budget_overflow_blocks_delivery(self):
        # 极小额度：任何物化都超预算，alarm 不得交付。
        engine = ExecutionEngine(ModelPackage(base_raw()), budget_bytes=1)
        for event in base_raw()["timeline"]:
            engine.advance(event)
        alarm = [d for d in engine.delivered if d[0] == "alarm"]
        self.assertEqual(alarm, [])
        self.assertTrue(any(
            d.head == "alarm" and d.status == "pending" for d in engine.ledger))

    def test_peak_workspace_within_budget_on_success(self):
        raw = base_raw()
        engine = ExecutionEngine(ModelPackage(raw), raw["budget_bytes"])
        engine.run_timeline(raw["timeline"])
        self.assertLessEqual(
            engine.report()["工作区峰值字节"], raw["budget_bytes"])

    def test_numeric_cache_invalidated_on_new_revision(self):
        raw = base_raw()
        engine = ExecutionEngine(ModelPackage(raw), raw["budget_bytes"])
        engine.advance(["compressed_only"])
        engine.advance(["request", "trend"])
        self.assertTrue(engine._shared_cache)
        engine.replace_package(ModelPackage(copy.deepcopy(raw)))
        self.assertEqual(engine._shared_cache, {})

    def test_decision_reused_but_not_numerics_across_requests(self):
        raw = base_raw()
        engine = ExecutionEngine(ModelPackage(raw), raw["budget_bytes"])
        engine.advance(["compressed_only"])
        engine.advance(["request", "trend"])
        self.assertTrue(engine._shared_cache)
        fingerprints_before = set(engine._shared_cache)
        # 新模型包后再请求：数值缓存清空，之后按新包重新物化。
        engine.replace_package(ModelPackage(copy.deepcopy(raw)))
        self.assertEqual(engine._shared_cache, {})
        engine.advance(["request", "trend"])
        self.assertTrue(engine._shared_cache)
        for fp in engine._shared_cache:
            self.assertEqual(fp[0], "p7")
        # 决策缓存仍可基于结构条件重建（不携带旧数值）。
        self.assertTrue(engine._decision_cache)


class ReproducibilityTests(unittest.TestCase):
    def test_timeline_reorder_does_not_change_deterministic_outcome(self):
        raw = base_raw()
        # 送达顺序重排（请求仍在资料到达之后），最终交付集合应一致。
        timeline = [
            ["compressed_only"],
            ["arrive_original", "coupler"],
            ["arrive_original", "input"],
            ["request", "trend"],
            ["request", "alarm"],
        ]
        engine = ExecutionEngine(ModelPackage(raw), raw["budget_bytes"])
        engine.run_timeline(timeline)
        delivered = sorted((h, i) for (h, i) in engine.delivered)
        self.assertIn(("alarm", 0), delivered)
        self.assertIn(("alarm", 1), delivered)

    def test_repeated_runs_identical_reports(self):
        from tensor_exec.runner import run_scene
        r1 = run_scene(SCENE)
        r2 = run_scene(SCENE)
        self.assertEqual(
            json.dumps(r1, ensure_ascii=False, sort_keys=True),
            json.dumps(r2, ensure_ascii=False, sort_keys=True))

    def test_custom_slice_via_public_api(self):
        """调用者换轴切片后重跑，结果落在请求区域内。"""
        pkg = load(SCENE)
        head = pkg.heads["alarm"]
        sl = Slice({"channel": (1, 2), "batch": (1, 2)})
        # 未送达原值，使用压缩视图即可证明该局部坐标。
        result = execute_head(head, pkg, sl, 0, False)
        self.assertTrue(result.provable)
        self.assertEqual(result.center.shape, (1, 1))

    def test_fp64_output_has_tighter_bound_than_fp32(self):
        raw = base_raw()
        raw["heads"]["alarm"]["dtype"] = "fp64"
        pkg64 = ModelPackage(raw)
        r64 = execute_head(pkg64.heads["alarm"], pkg64,
                           pkg64.heads["alarm"].slices[0], 0, False)
        pkg32 = load(SCENE)
        r32 = execute_head(pkg32.heads["alarm"], pkg32,
                           pkg32.heads["alarm"].slices[0], 0, False)
        self.assertEqual(r64.center.dtype, np.float64)
        self.assertLessEqual(r64.max_error(), r32.max_error())


if __name__ == "__main__":
    unittest.main(verbosity=2)
