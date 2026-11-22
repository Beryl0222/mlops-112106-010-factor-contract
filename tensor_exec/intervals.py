"""公开误差传播契约。

每个因子视图都带 *逐元素绝对误差上界*：原值视图为 0，压缩视图来自交换
文件的块级 ``error``。对每个输出坐标，收缩维护四个量：

1. 中心值之和（FP64 累积）；
2. 最坏情形量级之和 ``Σ Π(|c_i| + e_i)``；
3. 中心量级之和 ``Σ Π|c_i|``；
4. 实际参与的乘积项数，用于估计累积舍入。

输入表示误差的最终上界为 ``最坏量级和 − 中心量级和``，它在数学上等价于
对完整乘积做包含全部二阶/高阶交叉项的区间展开，并且与切块粒度无关——
先裁剪再分块求和也能得到同样的界。该界不把“输出值很小”当作合格依据：
符号相消时中心值可能接近零，但误差界仍如实放大，由头的 ``error_limit``
裁决。
"""

from __future__ import annotations

import numpy as np

from .blocks import MACHINE_EPS


def term_product(centers: list[np.ndarray], errors: list[np.ndarray]):
    """一个乘积项的中心值、``Π|c|`` 与 ``Π(|c|+e)``（广播到同一形状）。"""
    shape = np.broadcast_shapes(*(c.shape for c in centers))
    center = np.ones(shape, dtype=np.float64)
    abs_product = np.ones(shape, dtype=np.float64)
    worst_product = np.ones(shape, dtype=np.float64)
    for c, e in zip(centers, errors, strict=True):
        c = np.broadcast_to(np.asarray(c, dtype=np.float64), shape)
        e = np.broadcast_to(np.asarray(e, dtype=np.float64), shape)
        center = center * c
        abs_product = abs_product * np.abs(c)
        worst_product = worst_product * (np.abs(c) + e)
    return center, abs_product, worst_product


def term_from_magnitudes(center: np.ndarray, abs_mag: np.ndarray,
                         worst_mag: np.ndarray,
                         extra_centers: list[np.ndarray],
                         extra_errors: list[np.ndarray]):
    """与已带聚合误差界的中间张量继续相乘。

    共享中间量已经给出每个坐标的中心值 ``center``、中心量级 ``abs_mag``
    与最坏量级 ``worst_mag``（后者已内含其输入误差与累积舍入）。把它整体
    当作乘积的一个因子，再叠加其余私有因子的逐元素误差，仍返回
    ``(中心值, Π|c|, Π(|c|+e))``。
    """
    shape = np.broadcast_shapes(center.shape,
                                *(c.shape for c in extra_centers))
    center = np.broadcast_to(np.asarray(center, dtype=np.float64), shape)
    abs_product = np.broadcast_to(np.asarray(abs_mag, dtype=np.float64), shape)
    worst_product = np.broadcast_to(np.asarray(worst_mag, dtype=np.float64),
                                    shape)
    out_center = np.array(center, copy=True)
    for c, e in zip(extra_centers, extra_errors, strict=True):
        c = np.broadcast_to(np.asarray(c, dtype=np.float64), shape)
        e = np.broadcast_to(np.asarray(e, dtype=np.float64), shape)
        out_center = out_center * c
        abs_product = abs_product * np.abs(c)
        worst_product = worst_product * (np.abs(c) + e)
    return out_center, abs_product, worst_product


class OutputAccumulator:
    """单个输出区域的收缩累积器。"""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.center = np.zeros(shape, dtype=np.float64)
        self.abs_sum = np.zeros(shape, dtype=np.float64)
        self.worst_sum = np.zeros(shape, dtype=np.float64)
        # 每个输出坐标实际累加的乘积项数（结构零跳过时不增加）。
        self.mac_terms = np.zeros(shape, dtype=np.int64)

    def add(self, term_center: np.ndarray, term_abs: np.ndarray,
            term_worst: np.ndarray) -> None:
        self.center += term_center
        self.abs_sum += term_abs
        self.worst_sum += term_worst
        self.mac_terms += np.ones_like(term_center, dtype=np.int64)

    def finalize(self, dtype: str):
        """关闭累积，返回 ``(中心值, 逐元素绝对误差上界)``。"""
        # 输入表示误差上界：最坏量级和减中心量级和（含全部交叉项）。
        input_bound = np.maximum(self.worst_sum - self.abs_sum, 0.0)
        u64 = float(MACHINE_EPS["fp64"])
        # 每次乘加至多引入 (k+1) 个机器精度的保守舍入界。
        acc_round = ((self.mac_terms.astype(np.float64) + 1.0) * u64
                     * self.abs_sum)
        bound = input_bound + acc_round
        center = self.center
        if dtype == "fp32":
            u32 = float(MACHINE_EPS["fp32"])
            # 最后一次向 FP32 舍入，按含误差的最坏量级估计。
            cast_round = u32 * (np.abs(center) + bound)
            bound = bound + cast_round
            center = center.astype(np.float32).astype(np.float64)
        elif dtype != "fp64":
            raise ValueError(f"不支持的输出精度 {dtype}")
        return center, bound

    def total_macs(self) -> int:
        """该输出区域实际执行的标量乘加总数（结构零不计）。"""
        return int(self.mac_terms.sum())
