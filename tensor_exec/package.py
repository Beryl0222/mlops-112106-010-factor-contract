"""模型包：命名轴、切块网格、头契约与资料到达状态。

模型包把交换 JSON 解释成按 *逻辑轴名* 寻址的结构。两个因子即使物理形状
相同，也只能通过逻辑轴名对齐——物理轴顺序在版本间可以调整（如 p8 中
``frequency`` 盘上轴序交换），逻辑含义不变，因此所有收缩都按轴名调度。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .blocks import Block, block_from_dict


@dataclass(frozen=True)
class Slice:
    """头的一个输出切片：被约束轴 -> 左闭右开区间，未列出的轴取全长。"""

    constraints: dict[str, tuple[int, int]]

    def region(self, head_axes: tuple[str, ...], lengths: dict[str, int]):
        """返回该切片在头输出轴上的逐轴区间。"""
        return tuple(self.constraints.get(a, (0, lengths[a])) for a in head_axes)


@dataclass(frozen=True)
class HeadSpec:
    """输出头契约：参与因子、保留逻辑轴、切片集合、误差上限与输出精度。"""

    name: str
    factors: tuple[str, ...]
    axes: tuple[str, ...]
    slices: tuple[Slice, ...]
    error_limit: float
    dtype: str

    @property
    def is_alarm(self) -> bool:
        return self.name == "alarm"


@dataclass
class FactorModel:
    """单个因子的逻辑结构与物理块集合。"""

    name: str
    physical_axes: tuple[str, ...]
    blocks: tuple[Block, ...]

    def logical_shape(self, lengths: dict[str, int]) -> tuple[int, ...]:
        return tuple(lengths[a] for a in self.physical_axes)


class ModelPackage:
    """一次有效模型包（固定 revision）。

    包负责：

    * 校验块区间、轴名与切片合法性；
    * 在逻辑轴网格上区分 *结构零块*、*有数据块* 与 *缺块*；
    * 跟踪各因子压缩视图与原值视图的实际到达状态。
    """

    def __init__(self, raw: dict):
        self.revision: str = str(raw["revision"])
        self.lengths: dict[str, int] = {k: int(v) for k, v in raw["lengths"].items()}
        self.factors: dict[str, FactorModel] = {}
        for name, spec in raw["factors"].items():
            axes = tuple(spec["axes"])
            for axis in axes:
                if axis not in self.lengths:
                    raise ValueError(f"因子 {name} 引用了未知逻辑轴 {axis}")
            if len(set(axes)) != len(axes):
                raise ValueError(f"因子 {name} 内逻辑轴名必须唯一")
            blocks = tuple(block_from_dict(name, axes, b) for b in spec["blocks"])
            self._validate_blocks(name, axes, blocks)
            self.factors[name] = FactorModel(name, axes, blocks)

        self.heads: dict[str, HeadSpec] = {}
        for name, spec in raw["heads"].items():
            self.heads[name] = self._parse_head(name, spec)

        # 到达状态：压缩视图随包即到；原值视图由时间线补交。
        self._original_arrived: set[str] = set()
        self.compressed_only = False

    # ---- 构造与校验 -------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelPackage":
        # 交换文件是不受信资料，调用方应以独立目录和 python -I 运行。
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw)

    def _validate_blocks(self, name, axes: tuple[str, ...], blocks: tuple[Block, ...]):
        for block in blocks:
            for axis, (start, stop) in zip(axes, block.at, strict=True):
                if not (0 <= start < stop <= self.lengths[axis]):
                    raise ValueError(
                        f"因子 {name} 块 {block.at} 超出逻辑轴 {axis} 长度"
                    )
        # 同一逻辑坐标最多被一个块覆盖：两个块只有在 *所有* 轴区间同时
        # 相交时才算冲突（多轴块允许在单个轴上共享区间）。
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                overlap = True
                for axis, r1, r2 in zip(
                    axes, blocks[i].at, blocks[j].at, strict=True
                ):
                    if max(r1[0], r2[0]) >= min(r1[1], r2[1]):
                        overlap = False
                        break
                if overlap:
                    raise ValueError(
                        f"因子 {name} 的块 {blocks[i].at} 与 {blocks[j].at} "
                        "在逻辑坐标上重叠"
                    )

    def _parse_head(self, name: str, spec: dict) -> HeadSpec:
        factors = tuple(spec["factors"])
        for factor in factors:
            if factor not in self.factors:
                raise ValueError(f"头 {name} 引用了未知因子 {factor}")
        axes = tuple(spec["axes"])
        if len(set(axes)) != len(axes):
            raise ValueError(f"头 {name} 的输出轴必须唯一")
        for axis in axes:
            if not any(axis in self.factors[f].physical_axes for f in factors):
                raise ValueError(f"头 {name} 的输出轴 {axis} 不被任何因子覆盖")
        slices = tuple(
            Slice({a: (int(lo), int(hi)) for a, (lo, hi) in s.items()})
            for s in spec["slices"]
        )
        for sl in slices:
            for axis, (lo, hi) in sl.constraints.items():
                if axis not in axes:
                    raise ValueError(f"头 {name} 切片约束了非输出轴 {axis}")
                if not (0 <= lo < hi <= self.lengths[axis]):
                    raise ValueError(f"头 {name} 切片 {axis}=[{lo},{hi}) 非法")
        dtype = str(spec.get("dtype", "fp32"))
        if dtype not in ("fp32", "fp64"):
            raise ValueError(f"头 {name} 请求了不支持的输出精度 {dtype}")
        return HeadSpec(
            name=name,
            factors=factors,
            axes=axes,
            slices=slices,
            error_limit=float(spec["error_limit"]),
            dtype=dtype,
        )

    # ---- 切块网格 ---------------------------------------------------

    def blocks_on(self, factor_name: str) -> tuple[Block, ...]:
        return self.factors[factor_name].blocks

    def covering_block(self, factor_name: str, logical_ranges) -> Block | None:
        """找到物理轴区间完全覆盖给定逻辑区间的块，没有则返回 ``None``（缺块）。"""
        model = self.factors[factor_name]
        for block in model.blocks:
            ok = True
            for axis, (lo, hi) in logical_ranges:
                if axis not in model.physical_axes:
                    continue
                idx = model.physical_axes.index(axis)
                b_lo, b_hi = block.at[idx]
                if not (b_lo <= lo and hi <= b_hi):
                    ok = False
                    break
            if ok:
                return block
        return None

    def grid_cells(self, factor_name: str):
        """枚举因子覆盖区域内的切块网格单元。

        生成 ``(逻辑区间元组, 块或 None)``；``None`` 表示缺块，
        结构零块以 ``block.zero`` 标记，二者语义不同。
        """
        model = self.factors[factor_name]
        # 以现有块的边界为切分点，得到轴上的网格坐标。
        cuts_per_axis = []
        for axis in model.physical_axes:
            idx = model.physical_axes.index(axis)
            cuts = {0, self.lengths[axis]}
            for block in model.blocks:
                cuts.add(block.at[idx][0])
                cuts.add(block.at[idx][1])
            cuts_per_axis.append(sorted(cuts))

        def walk(axis_idx, ranges):
            if axis_idx == len(model.physical_axes):
                logical = tuple((a, (lo, hi)) for a, lo, hi in ranges)
                block = self.covering_block(factor_name, logical)
                yield logical, block
                return
            axis = model.physical_axes[axis_idx]
            cuts = cuts_per_axis[axis_idx]
            for lo, hi in zip(cuts, cuts[1:], strict=False):
                ranges.append((axis, lo, hi))
                yield from walk(axis_idx + 1, ranges)
                ranges.pop()

        yield from walk(0, [])

    # ---- 资料到达状态 ----------------------------------------------

    def mark_compressed_only(self) -> None:
        """对同时存在两种视图的块暂不交接原值（axis-revision 约定）。"""
        self.compressed_only = True
        self._original_arrived.clear()

    def arrive_original(self, factor_name: str) -> None:
        if factor_name not in self.factors:
            raise ValueError(f"未知因子 {factor_name} 的原值送达")
        factor = self.factors[factor_name]
        if not any(b.has_original() for b in factor.blocks):
            raise ValueError(f"因子 {factor_name} 没有声明原值视图，无法补交")
        self._original_arrived.add(factor_name)

    def view_available(self, block: Block, prefer_original: bool) -> bool:
        """块当前是否有可用视图。结构零始终“可用”（无需数据）。

        到达规则：

        * 只有原值视图的块，其原值是包内固有数据，立即可用；
        * 同时有压缩与原值两种视图的块，``compressed_only`` 冻结期内原值
          不可用；该因子收到显式 ``arrive_original`` 后冻结解除，原值可用；
        * 请求压缩时压缩视图始终可用；缺失时回退到可用的原值
          （原值是误差为 0 的更优来源）。
        """
        if block.zero:
            return True
        original_ready = block.has_original() and (
            not block.has_compressed()
            or block.factor in self._original_arrived
        )
        if prefer_original:
            return original_ready
        if block.has_compressed():
            return True
        return original_ready

    def effective_view(self, factor_name: str) -> str:
        """该因子本次有效模型包下实际可消费的视图级别。"""
        factor = self.factors[factor_name]

        def original_ready(block: Block) -> bool:
            if block.zero:
                return True
            return block.has_original() and (
                not block.has_compressed()
                or factor_name in self._original_arrived
            )

        if all(original_ready(b) for b in factor.blocks) and all(
            b.zero or b.has_original() for b in factor.blocks
        ):
            return "original"
        if all(b.zero or b.has_compressed() for b in factor.blocks):
            return "compressed"
        if all(b.zero or b.has_compressed()
               or (b.has_original() and not b.has_compressed())
               for b in factor.blocks):
            return "compressed"
        return "incomplete"
