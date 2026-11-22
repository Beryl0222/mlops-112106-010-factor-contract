"""因子块数据模型：逻辑轴、物理切块、结构零与双视图。

每个 :class:`Block` 对应交换文件中一个 ``blocks`` 条目：

* ``at`` 给出各 *物理* 轴位置上的逻辑坐标区间（左闭右开）；
* ``zero: true`` 表示该块覆盖的逻辑坐标精确为零（结构零）；
* 原值视图是 FP32 精确值（误差允许为 0 的精确来源）；
* 压缩视图由整数中心 ``center``、比例 ``scale`` 与逐块逐元素绝对误差
  上界 ``error`` 组成，中心值按 ``scale * center`` 解码。

缺块（轴长范围内没有任何条目覆盖的逻辑坐标）与结构零语义不同，
由 :class:`~tensor_exec.package.ModelPackage` 在切块网格上显式区分。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 公开精度对应的机器精度（fp32 累积/输出、fp64 累积）。
MACHINE_EPS = {"fp32": np.float32(2.0**-23), "fp64": np.float32(2.0**-52)}


@dataclass(frozen=True)
class Block:
    """单个物理切块的不可变描述。"""

    factor: str
    at: tuple[tuple[int, int], ...]
    physical_axes: tuple[str, ...]
    zero: bool = False
    center: np.ndarray | None = None
    scale: float | None = None
    error: float | None = None
    original: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """块在各物理轴上的尺寸。"""
        return tuple(stop - start for start, stop in self.at)

    def has_original(self) -> bool:
        return self.original is not None

    def has_compressed(self) -> bool:
        return self.center is not None

    def view(self, prefer_original: bool):
        """返回该块本次使用的视图。

        返回 ``(中心值数组, 逐元素绝对误差上界, 视图名)``。
        请求原值且已送达时使用原值（误差为 0）；请求压缩时压缩视图优先，
        若该块只有原值则回退原值。结构零由调用方先跳过，不会走到这里。
        """
        if prefer_original and self.original is not None:
            return np.asarray(self.original, dtype=np.float64), 0.0, "original"
        if self.center is not None:
            values = np.asarray(self.center, dtype=np.float64) * float(self.scale)
            return values, float(self.error or 0.0), "compressed"
        if self.original is not None:
            return np.asarray(self.original, dtype=np.float64), 0.0, "original"
        raise ValueError(f"因子 {self.factor} 的块 {self.at} 没有可用视图")

    def as_original(self) -> np.ndarray:
        if self.original is None:
            raise ValueError(f"因子 {self.factor} 的块 {self.at} 缺少原值视图")
        return np.asarray(self.original, dtype=np.float64)


def block_from_dict(factor_name: str, axes: tuple[str, ...], raw: dict) -> Block:
    """按交换文件条目构造 :class:`Block`，物理轴顺序以 factor 条目为准。"""
    at = tuple((int(start), int(stop)) for start, stop in raw["at"])
    if len(at) != len(axes):
        raise ValueError(
            f"因子 {factor_name} 的块 {at} 轴数与物理轴 {axes} 不一致"
        )
    shape = tuple(stop - start for start, stop in at)
    zero = bool(raw.get("zero", False))

    center = raw.get("center")
    if center is not None:
        center = np.asarray(center, dtype=np.float64)
        if tuple(center.shape) != shape:
            raise ValueError(f"因子 {factor_name} 压缩中心形状与切块 {at} 不符")
    original = raw.get("original")
    if original is not None:
        original = np.asarray(original, dtype=np.float64)
        if tuple(original.shape) != shape:
            raise ValueError(f"因子 {factor_name} 原值形状与切块 {at} 不符")
    if not zero and center is None and original is None:
        raise ValueError(
            f"因子 {factor_name} 的块 {at} 既非结构零也没有任何数值视图"
        )
    if zero and (center is not None or original is not None):
        raise ValueError(f"因子 {factor_name} 的结构零块 {at} 不得携带数值视图")

    return Block(
        factor=factor_name,
        at=at,
        physical_axes=tuple(axes),
        zero=zero,
        center=center,
        scale=float(raw["scale"]) if "scale" in raw else None,
        error=float(raw["error"]) if "error" in raw else None,
        original=original,
    )
