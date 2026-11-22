"""块级张量收缩原语。

收缩严格按 *逻辑轴名* 对齐：因子物理数组先转置到该头的规范轴布局
（输出轴 + 求和轴），缺轴补长度 1 的广播维。因此两个同形因子不会因为
物理顺序相同就被错误对轴——p8 中 ``frequency`` 物理轴序交换后数值结果
必须与 p7 一致。

执行方式是逐块流式的：枚举各因子在网格上的单元组合，求和轴区间一致的
组合才参与一次乘加；结构零块整体跳过（零次乘加），缺块使对应输出坐标
不可证，绝不以零冒充。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .blocks import MACHINE_EPS, Block
from .intervals import (
    OutputAccumulator,
    term_from_magnitudes,
    term_product,
)
from .package import ModelPackage, Slice


@dataclass
class MissingRequirement:
    """一块缺失证据：哪些逻辑坐标因为缺块或视图未到而不可证。"""

    factor: str
    logical_coords: dict[str, tuple[int, int]]
    reason: str


@dataclass
class HeadResult:
    """一个切片一次收缩尝试的完整结果。"""

    head: str
    slice_index: int
    provable: bool
    center: np.ndarray | None = None
    bound: np.ndarray | None = None
    dtype: str = "fp32"
    macs: int = 0
    views_used: dict[str, str] = field(default_factory=dict)
    missing: list[MissingRequirement] = field(default_factory=list)
    used_shared: bool = False
    workspace_bytes: int = 0
    reads_bytes: int = 0
    # 共享路径下本切片对应的共享阶段乘加（批次内只实际执行一次）。
    shared_stage_macs: int = 0

    def max_error(self) -> float | None:
        return None if self.bound is None else float(np.max(self.bound))

    def within_limit(self, error_limit: float) -> bool:
        if self.bound is None:
            return False
        # 与头契约比较用的是误差上界，输出值本身不参与合格判断。
        return bool(np.all(self.bound <= error_limit))


@dataclass
class SharedIntermediate:
    """多头公共子表达式的物化中间张量。"""

    factors: tuple[str, ...]
    axes: tuple[str, ...]  # 保留的逻辑轴（求和轴已消去）
    center: np.ndarray
    bound: np.ndarray
    abs_mag: np.ndarray  # 中心量级和 ΣΠ|c|
    worst_mag: np.ndarray  # 最坏量级和 ΣΠ(|c|+e)，已含舍入
    macs: int
    views_used: dict[str, str]
    fingerprint: tuple  # 复用判定键（不含可复用决策之外的数值版本信息）

    @property
    def nbytes(self) -> int:
        return int(self.center.nbytes + self.bound.nbytes
                   + self.abs_mag.nbytes + self.worst_mag.nbytes)


def head_axes_universe(head, package: ModelPackage) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """返回头的 ``(输出轴, 求和轴)``；求和轴按名排序保证可重复。"""
    used = set()
    for fname in head.factors:
        used.update(package.factors[fname].physical_axes)
    summed = tuple(sorted(used - set(head.axes)))
    return tuple(head.axes), summed


def _slice_region(head, package: ModelPackage, sl: Slice):
    return sl.region(head.axes, package.lengths)


def _canonical_array(block: Block, model_axes, universe: tuple[str, ...],
                     values: np.ndarray, error: float):
    """把物理布局的块数组转成头规范轴布局，缺轴补广播维。"""
    # 物理轴 -> 逻辑轴的映射由 factor 条目保证，逐轴转置。
    perm = [model_axes.index(a) for a in universe if a in model_axes]
    values = np.transpose(values, perm)
    present = tuple(a for a in universe if a in model_axes)
    out_shape = tuple(
        values.shape[present.index(a)] if a in model_axes else 1
        for a in universe
    )
    values = values.reshape(out_shape)
    err = np.full(out_shape, float(error), dtype=np.float64)
    return values, err


def _materialize(block: Block | None, package: ModelPackage, prefer_original: bool,
                 factor_name: str = "?"):
    """物化一个块，返回 ``(中心, 误差, 视图名, 字节数)`` 或缺块原因。"""
    if block is None:
        return MissingRequirement(
            factor_name, {}, "缺块：该逻辑坐标没有任何数据，不能冒充零")
    if block.zero:
        shape = block.shape
        zeros = np.zeros(shape, dtype=np.float64)
        return zeros, np.zeros(shape, dtype=np.float64), "structural_zero", 0
    if not package.view_available(block, prefer_original):
        wanted = "原值" if prefer_original else "压缩"
        return MissingRequirement(
            block.factor,
            {a: lo_hi for a, lo_hi in zip(block.physical_axes, block.at, strict=True)},
            f"{wanted}视图尚未送达",
        )
    values, error, view_name = block.view(prefer_original)
    nbytes = int(values.size * 8 * 2)  # 中心与误差各一份 FP64 工作区
    return values, np.full(block.shape, error, dtype=np.float64), view_name, nbytes


def required_cell_tuples(head, package: ModelPackage, sl: Slice):
    """枚举切片区域内所有因子块单元组合（仅保留求和轴相容的组合）。

    输出轴允许只取块的一个子区域：与切片区间相交的块保留，实际裁剪在
    物化阶段按逻辑坐标完成；缺块（block 为 None）同样保留以暴露不可证。
    """
    output_axes, summed = head_axes_universe(head, package)
    region_by_axis = dict(zip(output_axes, _slice_region(head, package, sl),
                              strict=True))

    per_factor_cells = []
    for fname in head.factors:
        cells = []
        for ranges, block in package.grid_cells(fname):
            ranges_by_axis = dict(ranges)
            # 输出轴：与切片区间有交集即可（块可能比切片大）；求和轴全保留。
            disjoint = any(
                axis in region_by_axis
                and (cell_hi <= region_by_axis[axis][0]
                     or cell_lo >= region_by_axis[axis][1])
                for axis, (cell_lo, cell_hi) in ranges
            )
            if disjoint:
                continue
            cells.append((ranges_by_axis, block))
        per_factor_cells.append(cells)

    for combo in product(*per_factor_cells):
        # 同一求和轴允许不同因子的切块粒度不同：各因子区间只要互相有交集
        # 就按交集收缩（细块可落在粗块内部）。逐轴求所有因子的公共交集。
        axis_intervals: dict[str, list[tuple[int, int]]] = {}
        for ranges_by_axis, _block in combo:
            for axis, rng in ranges_by_axis.items():
                if axis in output_axes:
                    continue
                axis_intervals.setdefault(axis, []).append(rng)
        intersection: dict[str, tuple[int, int]] = {}
        compatible = True
        for axis, intervals in axis_intervals.items():
            lo = max(i[0] for i in intervals)
            hi = min(i[1] for i in intervals)
            if lo >= hi:
                compatible = False
                break
            intersection[axis] = (lo, hi)
        if compatible:
            yield combo, intersection


def _crop_to_effective(block: Block, model_axes, values: np.ndarray,
                       err: np.ndarray, ranges_by_axis, constraints,
                       output_axes):
    """按输出切片区间与求和轴交集对块做物理轴裁剪。

    ``constraints`` 给出每个需要约束的逻辑轴目标区间（左闭右开）。
    返回 ``(值, 误差, 裁剪后逻辑区间)``；块与约束的交集一定非空。
    """
    physical_slicer = []
    effective: dict[str, tuple[int, int]] = {}
    for axis in model_axes:
        lo, hi = ranges_by_axis[axis]
        if axis in constraints:
            c_lo, c_hi = constraints[axis]
            new_lo, new_hi = max(lo, c_lo), min(hi, c_hi)
            if new_lo >= new_hi:
                raise ValueError(f"块在轴 {axis} 上与约束区间无交集")
            physical_slicer.append(slice(new_lo - lo, new_hi - lo))
            effective[axis] = (new_lo, new_hi)
        else:
            physical_slicer.append(slice(None))
            effective[axis] = (lo, hi)
    return values[tuple(physical_slicer)], err[tuple(physical_slicer)], effective


def execute_head(head, package: ModelPackage, sl: Slice, slice_index: int,
                 prefer_original: bool) -> HeadResult:
    """对单个切片执行完整的通用收缩（无共享中间量）。"""
    output_axes, summed = head_axes_universe(head, package)
    region = _slice_region(head, package, sl)
    out_shape = tuple(hi - lo for lo, hi in region)
    acc = OutputAccumulator(out_shape)
    universe = output_axes + summed

    result = HeadResult(head=head.name, slice_index=slice_index, provable=True,
                        dtype=head.dtype)
    workspace_peak = 0
    reads = 0
    views: dict[str, set[str]] = {}
    region_by_axis = dict(zip(output_axes, region, strict=True))

    for combo, intersection in required_cell_tuples(head, package, sl):
        # 结构零先判：任一因子块为结构零，整个乘积项精确为零。
        if any(block is not None and block.zero for _r, block in combo):
            continue
        # 约束 = 输出切片区间 + 求和轴公共交集。
        constraints = dict(intersection)
        constraints.update(region_by_axis)
        mats = []
        out_slices = []
        local_peak = 0
        abort = False
        for (ranges_by_axis, block), fname in zip(combo, head.factors, strict=True):
            mat = _materialize(block, package, prefer_original, fname)
            if isinstance(mat, MissingRequirement):
                result.provable = False
                result.missing.append(mat)
                abort = True
                break
            values, err, view_name, nbytes = mat
            local_peak += nbytes
            reads += nbytes
            views.setdefault(fname, set()).add(view_name)
            model_axes = package.factors[fname].physical_axes
            # 按输出区间与求和轴交集裁出有效子区域（含误差数组）。
            values, err, effective = _crop_to_effective(
                block, model_axes, values, err, ranges_by_axis,
                constraints, output_axes)
            c, e = _canonical_array(block, model_axes, universe, values,
                                    0.0 if block.zero else float(err.flat[0]))
            mats.append((c, e))
            # 输出轴上的写入位置（基于裁剪后的有效逻辑区间）。
            idx = []
            for axis, (base_lo, _base_hi) in zip(output_axes, region, strict=True):
                if axis in effective:
                    c_lo, c_hi = effective[axis]
                    idx.append(slice(c_lo - base_lo, c_hi - base_lo))
                else:
                    idx.append(slice(None))
            out_slices.append(tuple(idx))
        if abort:
            continue
        centers = [m[0] for m in mats]
        errors = [m[1] for m in mats]
        term_center, term_abs, term_worst = term_product(centers, errors)
        if summed:
            sum_axes = tuple(range(len(output_axes), len(universe)))
            term_center = term_center.sum(axis=sum_axes)
            term_abs = term_abs.sum(axis=sum_axes)
            term_worst = term_worst.sum(axis=sum_axes)
        target = out_slices[0]
        acc.add(
            np.broadcast_to(term_center, acc.center[target].shape).copy(),
            np.broadcast_to(term_abs, acc.abs_sum[target].shape).copy(),
            np.broadcast_to(term_worst, acc.worst_sum[target].shape).copy())
        workspace_peak = max(workspace_peak, local_peak)

    # 工作区还需容纳输出累积器本身（中心+两个量级和，FP64）。
    result.workspace_bytes = workspace_peak + int(
        acc.center.nbytes + acc.abs_sum.nbytes + acc.worst_sum.nbytes)
    result.reads_bytes = reads
    result.macs = acc.total_macs()
    result.views_used = {f: sorted(v) for f, v in views.items()}

    if not result.provable:
        return result
    center, bound = acc.finalize(head.dtype)
    result.center, result.bound = center, bound
    return result


# ---- 多头公共子表达式 ----------------------------------------------

def common_factor_set(heads) -> tuple[str, ...]:
    """所有活动头共同引用的因子（按首次出现顺序，保证可重复）。"""
    if not heads:
        return ()
    common = set(heads[0].factors)
    for head in heads[1:]:
        common &= set(head.factors)
    return tuple(f for f in heads[0].factors if f in common)


def shared_plan(heads, package: ModelPackage, prefer_original: bool):
    """推导共享计划的结构信息与缓存指纹（不执行任何数值计算）。

    返回 ``(指纹, 公共因子, 保留轴, 提前消去轴)`` 或 ``None``。
    """
    common = common_factor_set(heads)
    if len(common) < 2:
        return None
    common_axes = set()
    for fname in common:
        common_axes.update(package.factors[fname].physical_axes)
    head_output_axes: set[str] = set()
    head_summed_axes: set[str] = set()
    private_axes: set[str] = set()
    for head in heads:
        out_axes, summed = head_axes_universe(head, package)
        head_output_axes.update(out_axes)
        head_summed_axes.update(summed)
        for fname in head.factors:
            if fname not in common:
                private_axes.update(package.factors[fname].physical_axes)
    shared_summed = tuple(sorted(
        common_axes & head_summed_axes
        - head_output_axes - private_axes
    ))
    kept_axes = tuple(sorted(common_axes - set(shared_summed)))
    if not shared_summed:
        return None
    view_level = "original" if prefer_original else "compressed"
    # 视图级别逐因子记录，修订+轴+视图级别共同决定数值是否可复用。
    views = tuple(sorted(
        (f, package.effective_view(f)) for f in common
    ))
    fingerprint = (
        package.revision, common, kept_axes, shared_summed, views, view_level,
    )
    return fingerprint, common, kept_axes, shared_summed


def build_shared(heads, package: ModelPackage, prefer_original: bool
                 ) -> tuple[SharedIntermediate | None, list[MissingRequirement]]:
    """构造公共子表达式：对公共因子求和“所有头都消去”的轴。"""
    plan = shared_plan(heads, package, prefer_original)
    missing: list[MissingRequirement] = []
    if plan is None:
        return None, missing
    fingerprint, common, kept_axes, shared_summed = plan

    universe = kept_axes + shared_summed
    out_shape = tuple(package.lengths[a] for a in kept_axes)
    acc = OutputAccumulator(out_shape)
    views: dict[str, set[str]] = {}
    reads = 0

    cells_per_factor = []
    for fname in common:
        cells_per_factor.append(list(package.grid_cells(fname)))
    for combo in product(*cells_per_factor):
        # 求和轴取所有公共因子的交集；保留轴（如共享批轴）必须区间一致。
        sum_intervals: dict[str, list[tuple[int, int]]] = {}
        kept_ranges: dict[str, tuple[int, int]] = {}
        compatible = True
        for ranges, _block in combo:
            for axis, rng in ranges:
                if axis in shared_summed:
                    sum_intervals.setdefault(axis, []).append(rng)
                else:
                    if axis in kept_ranges and kept_ranges[axis] != rng:
                        compatible = False
                    kept_ranges[axis] = rng
        if not compatible:
            continue
        intersection: dict[str, tuple[int, int]] = {}
        for axis, intervals in sum_intervals.items():
            lo, hi = max(i[0] for i in intervals), min(i[1] for i in intervals)
            if lo >= hi:
                compatible = False
                break
            intersection[axis] = (lo, hi)
        if not compatible:
            continue
        if any(block is None for _r, block in combo):
            for (ranges, block), fname in zip(combo, common, strict=True):
                if block is None:
                    missing.append(MissingRequirement(
                        fname, {a: rng for a, rng in ranges},
                        "公共子表达式遇到缺块"))
            continue
        if any(block is not None and block.zero for _r, block in combo):
            continue
        mats = []
        effective_kept: dict[str, tuple[int, int]] = {}
        for (ranges, block), fname in zip(combo, common, strict=True):
            mat = _materialize(block, package, prefer_original, fname)
            if isinstance(mat, MissingRequirement):
                missing.append(mat)
                compatible = False
                break
            values, err, view_name, nbytes = mat
            reads += nbytes
            views.setdefault(fname, set()).add(view_name)
            model_axes = package.factors[fname].physical_axes
            # 仅按求和轴交集裁剪；保留轴保持块原样（区间已验证一致）。
            values, err, eff = _crop_to_effective(
                block, model_axes, values, err, dict(ranges),
                intersection, output_axes=())
            effective_kept.update(
                (a, r) for a, r in eff.items() if a in kept_axes)
            c, e = _canonical_array(block, model_axes, universe, values,
                                    float(err.flat[0]))
            mats.append((c, e))
        if not compatible:
            continue
        term_center, term_abs, term_worst = term_product(
            [m[0] for m in mats], [m[1] for m in mats])
        sum_axes = tuple(range(len(kept_axes), len(universe)))
        term_center = term_center.sum(axis=sum_axes)
        term_abs = term_abs.sum(axis=sum_axes)
        term_worst = term_worst.sum(axis=sum_axes)
        # 保留轴上的写入位置（各因子保留轴区间一致）。
        target = tuple(
            slice(effective_kept[a][0], effective_kept[a][1])
            if a in effective_kept else slice(None)
            for a in kept_axes
        )
        acc.add(
            np.broadcast_to(term_center, acc.center[target].shape).copy(),
            np.broadcast_to(term_abs, acc.abs_sum[target].shape).copy(),
            np.broadcast_to(term_worst, acc.worst_sum[target].shape).copy())

    if missing:
        return None, missing
    # 共享阶段输入误差界（含全部交叉项）与 FP64 累积舍入；
    # 中间张量保持 FP64，不做输出精度转换。
    input_bound = np.maximum(acc.worst_sum - acc.abs_sum, 0.0)
    u64 = float(MACHINE_EPS["fp64"])
    round_bound = ((acc.mac_terms.astype(np.float64) + 1.0) * u64
                   * acc.abs_sum)
    center = acc.center
    bound = input_bound + round_bound
    # 最坏量级并入舍入，供私有阶段当作整体因子继续传播。
    worst_mag = acc.worst_sum + round_bound
    return SharedIntermediate(
        factors=common, axes=kept_axes, center=center, bound=bound,
        abs_mag=acc.abs_sum, worst_mag=worst_mag,
        macs=acc.total_macs(),
        views_used={f: sorted(v) for f, v in views.items()},
        fingerprint=fingerprint,
    ), missing


def apply_private(head, package: ModelPackage, sl: Slice, slice_index: int,
                  shared: SharedIntermediate, prefer_original: bool = True,
                  shared_reused: bool = False) -> HeadResult:
    """在共享中间量上叠加头的私有因子，完成该头切片的收缩。

    共享张量按 :attr:`SharedIntermediate.axes` 的逻辑轴名对齐，与因子的
    物理布局无关；``shared_reused`` 表示该中间量来自缓存（本次不重复
    计费共享阶段的乘加与读取）。
    """
    output_axes, summed = head_axes_universe(head, package)
    region = _slice_region(head, package, sl)
    out_shape = tuple(hi - lo for lo, hi in region)
    acc = OutputAccumulator(out_shape)
    universe = output_axes + summed
    result = HeadResult(head=head.name, slice_index=slice_index, provable=True,
                        dtype=head.dtype, used_shared=True)
    views: dict[str, set[str]] = {
        f: {f"shared:{v}"} for f, v in shared.views_used.items()
    }
    reads = 0

    private_factors = tuple(f for f in head.factors if f not in shared.factors)
    private_cells = [list(package.grid_cells(f)) for f in private_factors]

    def reshape_shared(array: np.ndarray, ranges_by_axis):
        """把按 shared.axes 布局的张量重塑到本头 universe 布局（按轴名搬运）。

        共享张量没有的 universe 轴补长度 1 的广播维；全程只按逻辑轴名定位，
        物理轴序变化（如 p8）不会造成错轴。
        """
        shape = tuple(
            (ranges_by_axis[a][1] - ranges_by_axis[a][0]) if a in ranges_by_axis
            else package.lengths[a]
            for a in shared.axes
        )
        array = array.reshape(shape)
        # 依次在 universe 位置上插入缺失轴，已存在的轴随插入自然右移。
        for pos, axis in enumerate(universe):
            if axis not in shared.axes:
                array = np.expand_dims(array, axis=pos)
        out_shape_u = tuple(
            array.shape[universe.index(a)] if a in shared.axes else 1
            for a in universe
        )
        return array.reshape(out_shape_u)

    region_by_axis = dict(zip(output_axes, region, strict=True))

    for combo in product(*private_cells):
        ranges_by_axis: dict[str, tuple[int, int]] = {}
        # 求和轴交集：私有块之间、私有块与共享保留轴之间取公共区间。
        axis_intervals: dict[str, list[tuple[int, int]]] = {}
        for ranges, _block in combo:
            for axis, rng in ranges:
                ranges_by_axis[axis] = rng
                if axis not in output_axes:
                    axis_intervals.setdefault(axis, []).append(rng)
        for axis in shared.axes:
            if axis in summed:
                axis_intervals.setdefault(axis, []).append(
                    ranges_by_axis.get(axis, (0, package.lengths[axis])))
        intersection: dict[str, tuple[int, int]] = {}
        skip = False
        for axis, intervals in axis_intervals.items():
            lo, hi = max(i[0] for i in intervals), min(i[1] for i in intervals)
            if lo >= hi:
                skip = True
                break
            intersection[axis] = (lo, hi)
        if skip:
            continue
        # 输出轴：与切片区间不相交才跳过，部分相交走块内裁剪。
        if any(
            axis in region_by_axis
            and (rng[1] <= region_by_axis[axis][0]
                 or rng[0] >= region_by_axis[axis][1])
            for ranges, _b in combo for axis, rng in ranges
            if axis in output_axes
        ):
            continue
        if any(block is None for _r, block in combo):
            for (ranges, block), fname in zip(combo, private_factors, strict=True):
                if block is None:
                    result.provable = False
                    result.missing.append(MissingRequirement(
                        fname, {a: rng for a, rng in ranges}, "私有因子缺块"))
            continue
        if any(block is not None and block.zero for _r, block in combo):
            # 结构零私有块：整个项精确为零，跳过乘加。
            continue

        # 从共享张量中按本组合的轴区间切片，并重塑到 universe 布局。
        shared_idx = tuple(
            slice(*intersection[a]) if a in intersection
            else slice(*ranges_by_axis[a]) if a in ranges_by_axis
            else slice(None)
            for a in shared.axes
        )
        s_center = reshape_shared(shared.center[shared_idx], ranges_by_axis)
        s_abs = reshape_shared(shared.abs_mag[shared_idx], ranges_by_axis)
        s_worst = reshape_shared(shared.worst_mag[shared_idx], ranges_by_axis)

        constraints = dict(intersection)
        constraints.update(region_by_axis)
        extra_centers: list[np.ndarray] = []
        extra_errors: list[np.ndarray] = []
        effective: dict[str, tuple[int, int]] = {}
        abort = False
        for (ranges, block), fname in zip(combo, private_factors, strict=True):
            # 私有因子按自身到达状态独立选择最优可用视图：即使共享中间量
            # 仍是压缩的，已补齐原值的私有因子也应使用误差为 0 的原值。
            factor_prefers_original = (
                package.effective_view(fname) == "original")
            mat = _materialize(block, package, factor_prefers_original, fname)
            if isinstance(mat, MissingRequirement):
                result.provable = False
                result.missing.append(mat)
                abort = True
                break
            values, err, view_name, nbytes = mat
            reads += nbytes
            views.setdefault(fname, set()).add(view_name)
            model_axes = package.factors[fname].physical_axes
            values, err, eff = _crop_to_effective(
                block, model_axes, values, err, dict(ranges),
                constraints, output_axes)
            effective.update(eff)
            c, e = _canonical_array(block, model_axes, universe, values,
                                    float(err.flat[0]))
            extra_centers.append(c)
            extra_errors.append(e)
        if abort:
            continue

        term_center, term_abs, term_worst = term_from_magnitudes(
            s_center, s_abs, s_worst, extra_centers, extra_errors)
        sum_axes = tuple(i for i, a in enumerate(universe) if a in summed)
        if sum_axes:
            term_center = term_center.sum(axis=sum_axes)
            term_abs = term_abs.sum(axis=sum_axes)
            term_worst = term_worst.sum(axis=sum_axes)
        target = tuple(
            slice(effective[a][0] - base[0], effective[a][1] - base[0])
            if a in effective else slice(None)
            for a, base in zip(output_axes, region, strict=True)
        )
        acc.add(
            np.broadcast_to(term_center, acc.center[target].shape).copy(),
            np.broadcast_to(term_abs, acc.abs_sum[target].shape).copy(),
            np.broadcast_to(term_worst, acc.worst_sum[target].shape).copy())

    result.views_used = {f: sorted(v) for f, v in views.items()}
    # 切片只登记本头私有阶段实际新做的乘加与读取；共享阶段由引擎在批次
    # 级登记一次（跨头复用），避免把同一份公共收缩重复计费到每个头。
    result.macs = acc.total_macs()
    result.reads_bytes = reads
    result.workspace_bytes = shared.nbytes + int(
        acc.center.nbytes + acc.abs_sum.nbytes + acc.worst_sum.nbytes)
    result.shared_stage_macs = 0 if shared_reused else shared.macs
    if not result.provable:
        return result
    center, bound = acc.finalize(head.dtype)
    result.center, result.bound = center, bound
    return result
