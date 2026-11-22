"""张量收缩执行引擎：时间线驱动的调度、共享裁决与交付账本。

引擎在固定的工作区额度（``budget_bytes``）内同时约束 *物化量* 与
*跨输出共享占用*：

* **独立方案**：每个头各自跑通用收缩，峰值为各头工作区之和；
* **共享方案**：先物化公共因子的共享中间张量，再让各头叠加私有因子，
  共享量只物化一次，重复乘加与重复读取实际减少。

方案选择是可复用的 *执行决策*；而数值中间量只在同一模型包修订、同一
视图级别下复用。新模型包或不同输出请求到达时，旧数值一律失效。

交付语义：

* ``alarm`` 的全部切片必须同生共死——覆盖与误差全部成立才整体交付，
  任何一片未就绪都不会先暴露部分结果，等候中的切片仍计入工作区；
* ``trend`` 允许逐切片独立交付，每个交付坐标都记录所用包修订与该坐标
  实际达到的误差界。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .contraction import (
    HeadResult,
    SharedIntermediate,
    apply_private,
    build_shared,
    common_factor_set,
    execute_head,
    shared_plan,
)
from .package import ModelPackage


@dataclass
class Delivery:
    """一次交付/挂起的审计记录。"""

    event: int
    head: str
    slice_index: int
    status: str  # delivered | pending | rejected
    reason: str
    revision: str
    views_used: dict
    max_error: float | None
    error_limit: float
    macs: int
    workspace_bytes: int
    used_shared: bool
    shared_stage_macs: int = 0
    values: list | None = None

    def to_ledger(self) -> dict:
        d = {
            "事件序号": self.event,
            "头": self.head,
            "切片": self.slice_index,
            "状态": self.status,
            "原因": self.reason,
            "模型包修订": self.revision,
            "使用视图": self.views_used,
            "实际误差上界": self.max_error,
            "契约误差上限": self.error_limit,
            "私有乘加数": self.macs,
            "本批共享乘加(只执行一次)": self.shared_stage_macs,
            "工作区字节": self.workspace_bytes,
            "共享中间量": self.used_shared,
        }
        if self.values is not None:
            d["输出值"] = self.values
        return d


@dataclass
class _PlanChoice:
    kind: str  # shared | separate
    reason: str
    peak_bytes: int
    total_macs: int
    total_reads: int
    shared_saved_macs: int = 0
    shared_saved_reads: int = 0
    shared_stage_macs: int = 0


class ExecutionEngine:
    """按时间线推进收缩与交付的执行器。"""

    def __init__(self, package: ModelPackage, budget_bytes: int):
        self.package = package
        self.budget_bytes = int(budget_bytes)
        self.ledger: list[Delivery] = []
        self.delivered: dict[tuple[str, int], Delivery] = {}
        # 数值中间量缓存：fingerprint -> SharedIntermediate。
        self._shared_cache: dict[tuple, SharedIntermediate] = {}
        # 已承诺的执行决策（跨包可复用的只是决策，不是数值）。
        self._decision_cache: dict[tuple, _PlanChoice] = {}
        self._last_shared_missing: list = []
        self.event_index = -1

    # ---- 时间线 -----------------------------------------------------

    def run_timeline(self, timeline: list[list[str]]) -> list[Delivery]:
        for event in timeline:
            self.advance(event)
        return list(self.ledger)

    def advance(self, event: list[str]) -> list[Delivery]:
        """处理一条时间线事件，返回本事件产生的账本记录。"""
        self.event_index += 1
        before = len(self.ledger)
        kind = event[0]
        if kind == "compressed_only":
            self.package.mark_compressed_only()
            self._invalidate_numerics("进入 compressed_only，视图级别变化")
        elif kind == "arrive_original":
            self.package.arrive_original(event[1])
            self._invalidate_numerics(f"因子 {event[1]} 原值补齐，视图级别变化")
        elif kind == "new_package":
            # event: ["new_package", <新包 revision>]；调用方需先 replace_package。
            self._invalidate_numerics("新模型包生效，旧数值中间量全部失效")
        elif kind == "request":
            self._handle_request(event[1])
        else:
            raise ValueError(f"未知时间线事件 {event}")
        return self.ledger[before:]

    def replace_package(self, new_package: ModelPackage) -> None:
        """切换有效模型包；旧数值中间量作废，决策缓存按条件保留。"""
        old_rev = self.package.revision
        self.package = new_package
        self._invalidate_numerics(f"模型包修订 {old_rev} -> {new_package.revision}")

    def _invalidate_numerics(self, reason: str) -> None:
        self._shared_cache.clear()
        self._invalidate_reason = reason

    # ---- 请求处理 ---------------------------------------------------

    def _handle_request(self, head_name: str) -> None:
        head = self.package.heads[head_name]
        # 已交付切片幂等：重复请求不重新计算、不重复计费。
        if head.is_alarm and all(
            (head_name, i) in self.delivered for i in range(len(head.slices))
        ):
            self._record_idempotent(head_name)
            return
        if not head.is_alarm:
            pending_slices = [
                i for i in range(len(head.slices))
                if (head_name, i) not in self.delivered
            ]
            if not pending_slices:
                self._record_idempotent(head_name)
                return
        else:
            pending_slices = list(range(len(head.slices)))

        plan, results = self._plan_and_execute(head_name, pending_slices)
        self._account_and_deliver(head, pending_slices, results, plan)

    def _record_idempotent(self, head_name: str) -> None:
        for (h, idx), delivery in list(self.delivered.items()):
            if h != head_name:
                continue
            self.ledger.append(Delivery(
                event=self.event_index, head=h, slice_index=idx,
                status="delivered", reason="重复请求，沿用已交付结果（幂等）",
                revision=delivery.revision, views_used=delivery.views_used,
                max_error=delivery.max_error, error_limit=delivery.error_limit,
                macs=0, workspace_bytes=0, used_shared=delivery.used_shared,
            ))

    # ---- 方案裁决与执行 --------------------------------------------

    def _active_heads(self, requesting: str):
        """当前仍在等候交付的活动头（决定共享是否有消费方）。"""
        active = [self.package.heads[requesting]]
        for name, head in self.package.heads.items():
            if name == requesting:
                continue
            if head.is_alarm:
                if not all((name, i) in self.delivered
                           for i in range(len(head.slices))):
                    active.append(head)
            else:
                if any((name, i) not in self.delivered
                       for i in range(len(head.slices))):
                    active.append(head)
        return active

    def _plan_and_execute(self, head_name: str, pending_slices: list[int]):
        head = self.package.heads[head_name]
        prefer_original = self._prefer_original(head)
        active = self._active_heads(head_name)

        # 独立方案基线：每个待交付切片各自通用收缩。
        separate_results: list[HeadResult] = []
        separate_peak = 0
        separate_macs = 0
        separate_reads = 0
        for idx in pending_slices:
            r = execute_head(head, self.package, head.slices[idx], idx, prefer_original)
            separate_results.append(r)
            separate_peak += r.workspace_bytes
            separate_macs += r.macs
            separate_reads += r.reads_bytes
        separate_plan = _PlanChoice(
            "separate", "无可用共享或共享不满足额度/误差条件",
            separate_peak, separate_macs, separate_reads)

        # 共享方案：公共因子只在存在第二个活动消费方时才有意义。
        shared_plan_obj = None
        shared_results = None
        shared = None
        common = common_factor_set(active)
        plan_info = shared_plan(active, self.package, prefer_original)
        if len(active) >= 2 and plan_info is not None:
            fingerprint, common, _kept, _summed = plan_info
            cached = self._shared_cache.get(fingerprint)
            shared_reused = cached is not None
            if cached is not None:
                shared = cached
                reuse_note = "命中共享数值缓存，跳过共享阶段重复乘加与读取"
            else:
                built, missing = build_shared(active, self.package, prefer_original)
                if built is None:
                    # 共享覆盖不成立（缺块/视图未到）：记录原因后回退独立方案。
                    self._last_shared_missing = missing
                else:
                    shared = built
                    self._shared_cache[fingerprint] = shared
                    reuse_note = f"新建共享中间量；公共因子 {list(common)} 提前收缩"
            if shared is not None:
                candidate: list[HeadResult] = []
                peak = shared.nbytes  # 共享量在活动头工作期间常驻
                ok = True
                for idx in pending_slices:
                    r = apply_private(head, self.package, head.slices[idx],
                                      idx, shared,
                                      prefer_original=prefer_original,
                                      shared_reused=shared_reused)
                    candidate.append(r)
                    if not r.provable:
                        ok = False
                    peak += r.workspace_bytes - shared.nbytes
                if ok:
                    private_macs = sum(r.macs for r in candidate)
                    private_reads = sum(r.reads_bytes for r in candidate)
                    shared_macs_batch = 0 if shared_reused else shared.macs
                    shared_reads_batch = 0 if shared_reused else shared.nbytes
                    total_macs = private_macs + shared_macs_batch
                    total_reads = private_reads + shared_reads_batch
                    saved_macs = separate_macs - total_macs
                    saved_reads = separate_reads - total_reads
                    shared_plan_obj = _PlanChoice(
                        "shared", reuse_note,
                        peak, total_macs, total_reads,
                        max(saved_macs, 0), max(saved_reads, 0))
                    shared_plan_obj.shared_stage_macs = shared_macs_batch
                    shared_results = candidate
                else:
                    shared = None

        # 决策缓存键只描述结构条件；同结构条件可复用决策，不绑定数值。
        decision_key = (
            tuple(sorted(h.name for h in active)),
            tuple(common),
            "original" if prefer_original else "compressed",
            self.package.revision,
        )
        choice = self._choose(separate_plan, shared_plan_obj, decision_key)
        if choice.kind == "shared":
            return choice, shared_results
        return choice, separate_results

    def _choose(self, separate: _PlanChoice, shared: _PlanChoice | None,
                key: tuple) -> _PlanChoice:
        cached = self._decision_cache.get(key)
        if shared is None:
            chosen = separate
        elif shared.peak_bytes > self.budget_bytes and separate.peak_bytes <= self.budget_bytes:
            # 共享临时区不足：回退独立交付。
            chosen = _PlanChoice(
                "separate",
                f"共享方案需 {shared.peak_bytes} 字节，超过额度 {self.budget_bytes}，回退独立",
                separate.peak_bytes, separate.total_macs, separate.total_reads)
        elif separate.peak_bytes > self.budget_bytes and shared.peak_bytes <= self.budget_bytes:
            chosen = shared
        elif shared.peak_bytes > self.budget_bytes and separate.peak_bytes > self.budget_bytes:
            # 两个方案都超额度：选峰值较小者，但仍标记超限由交付裁决拒绝。
            chosen = shared if shared.peak_bytes <= separate.peak_bytes else separate
            chosen = replace(chosen, reason=chosen.reason + "；两方案均超工作区额度")
        elif shared.peak_bytes <= separate.peak_bytes:
            chosen = shared
        else:
            chosen = separate
        # 记录可复用决策；结构条件不变时下次直接沿用。
        self._decision_cache[key] = chosen
        if cached is not None and cached.kind == chosen.kind:
            chosen = replace(chosen, reason=chosen.reason + "（复用既有执行决策）")
        return chosen

    def _prefer_original(self, head) -> bool:
        """只有头的全部因子当前都有可用原值，才以原值视图收缩。"""
        for fname in head.factors:
            if self.package.effective_view(fname) != "original":
                return False
        return True

    # ---- 交付裁决与账本 --------------------------------------------

    def _account_and_deliver(self, head, pending_slices: list[int],
                             results: list[HeadResult], plan: _PlanChoice) -> None:
        limit = head.error_limit
        # 报警头：所有切片必须同时成立，否则整体挂起，不暴露部分结果。
        if head.is_alarm:
            all_provable = all(r.provable for r in results)
            all_within = all(r.within_limit(limit) for r in results)
            within_budget = plan.peak_bytes <= self.budget_bytes
            if all_provable and all_within and within_budget:
                for r in results:
                    self._deliver(head, r, plan, "报警组全部切片覆盖与误差成立，整体交付")
            else:
                for r in results:
                    reason = self._failure_reason(r, limit, plan)
                    self._pending(head, r, plan, reason)
            return

        # 趋势头：逐切片独立交付可证局部坐标。
        for r in results:
            if not r.provable:
                self._pending(head, r, plan, self._failure_reason(r, limit, plan))
            elif not r.within_limit(limit):
                self.ledger.append(Delivery(
                    event=self.event_index, head=head.name, slice_index=r.slice_index,
                    status="rejected",
                    reason=(f"实际误差上界 {r.max_error():.6g} 超过契约 "
                            f"{limit:.6g}（符号相消也不以小输出值放行）"),
                    revision=self.package.revision, views_used=r.views_used,
                    max_error=r.max_error(), error_limit=limit, macs=r.macs,
                    shared_stage_macs=(
                        plan.shared_stage_macs if r.used_shared else 0),
                    workspace_bytes=r.workspace_bytes,
                    used_shared=r.used_shared))
            elif plan.peak_bytes > self.budget_bytes:
                self._pending(head, r, plan,
                              f"工作区峰值 {plan.peak_bytes} 超过额度 {self.budget_bytes}")
            else:
                self._deliver(head, r, plan,
                              "趋势切片局部坐标可证且误差达标，独立交付")

    def _failure_reason(self, r: HeadResult, limit: float, plan: _PlanChoice) -> str:
        if not r.provable:
            details = "; ".join(
                f"因子 {m.factor} {m.logical_coords or '切片区域'}：{m.reason}"
                for m in r.missing) or "存在缺块或视图未送达"
            return f"覆盖不成立（缺块不能冒充零）：{details}"
        if not r.within_limit(limit):
            return (f"压缩误差放大后上界 {r.max_error():.6g} 超过契约 {limit:.6g}，"
                    "等待原值补齐")
        if plan.peak_bytes > self.budget_bytes:
            return f"工作区峰值 {plan.peak_bytes} 超过额度 {self.budget_bytes}"
        return "条件未全部成立"

    def _deliver(self, head, r: HeadResult, plan: _PlanChoice, reason: str) -> None:
        delivery = Delivery(
            event=self.event_index, head=head.name, slice_index=r.slice_index,
            status="delivered", reason=f"{reason}；方案：{plan.reason}",
            revision=self.package.revision, views_used=r.views_used,
            max_error=r.max_error(), error_limit=head.error_limit,
            macs=r.macs, workspace_bytes=plan.peak_bytes,
            used_shared=r.used_shared,
            shared_stage_macs=plan.shared_stage_macs if r.used_shared else 0,
            values=r.center.tolist() if r.center is not None else None)
        self.delivered[(head.name, r.slice_index)] = delivery
        self.ledger.append(delivery)

    def _pending(self, head, r: HeadResult, plan: _PlanChoice, reason: str) -> None:
        # 报警等候中的切片仍占工作区；趋势未交付不留存物化输出（不先还原全量）。
        held = plan.peak_bytes if head.is_alarm else r.workspace_bytes
        self.ledger.append(Delivery(
            event=self.event_index, head=head.name, slice_index=r.slice_index,
            status="pending", reason=reason, revision=self.package.revision,
            views_used=r.views_used, max_error=r.max_error(),
            error_limit=head.error_limit, macs=r.macs,
            shared_stage_macs=plan.shared_stage_macs if r.used_shared else 0,
            workspace_bytes=held, used_shared=r.used_shared))

    # ---- 报告 -------------------------------------------------------

    def report(self) -> dict:
        """产出可核对的审计报告（不含时间戳，保证可重复运行）。"""
        return {
            "模型包修订": self.package.revision,
            "工作区额度字节": self.budget_bytes,
            "工作区峰值字节": max(
                (d.workspace_bytes for d in self.ledger), default=0),
            "已交付": [
                d.to_ledger() for d in self.ledger if d.status == "delivered"
            ],
            "挂起": [d.to_ledger() for d in self.ledger if d.status == "pending"],
            "拒绝": [d.to_ledger() for d in self.ledger if d.status == "rejected"],
            "共享缓存条目": [
                {
                    "指纹": list(fp),
                    "因子": list(s.factors),
                    "保留轴": list(s.axes),
                    "字节": s.nbytes,
                    "乘加数": s.macs,
                }
                for fp, s in self._shared_cache.items()
            ],
            "决策缓存": [
                {"键": list(k), "方案": v.kind, "原因": v.reason,
                 "峰值字节": v.peak_bytes, "节省乘加": v.shared_saved_macs,
                 "节省读取字节": v.shared_saved_reads,
                 "本批共享乘加": v.shared_stage_macs}
                for k, v in self._decision_cache.items()
            ],
        }
