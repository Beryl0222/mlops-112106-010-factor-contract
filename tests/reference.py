"""暴力参考收缩：全量重建命名轴张量后用 einsum 收缩。

与执行器的块级流式实现相互独立，用于交叉验证。结构零坐标整体掩码，
缺块坐标在参考里直接暴露（测试中另行构造）。
"""

from __future__ import annotations

import numpy as np

from tensor_exec.package import ModelPackage


def dense(package: ModelPackage, fname: str, prefer_original: bool):
    fm = package.factors[fname]
    shape = tuple(package.lengths[a] for a in fm.physical_axes)
    center = np.zeros(shape)
    error = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    for block in fm.blocks:
        idx = tuple(slice(lo, hi) for lo, hi in block.at)
        if block.zero:
            mask[idx] = False  # 结构零：整项贡献精确为零（含误差），掩码排除
            continue
        values, err, _name = block.view(prefer_original)
        center[idx] = values
        error[idx] = err
        mask[idx] = True
    return center, error, mask, fm.physical_axes


def reference(package: ModelPackage, head_name: str, prefer_original: bool):
    head = package.heads[head_name]
    out = head.axes
    data = [dense(package, f, prefer_original) for f in head.factors]
    all_axes = set()
    for _c, _e, _m, axes in data:
        all_axes.update(axes)
    universe = tuple(out) + tuple(sorted(all_axes - set(out)))

    def to_universe(array, axes):
        """把物理轴序的数组对齐到 universe 布局，缺轴补长度 1 的维。"""
        # 先按 universe 中出现的物理轴顺序转置，再在缺失位置插入广播维。
        present = [a for a in universe if a in axes]
        perm = [axes.index(a) for a in present]
        array = np.transpose(array, perm)
        for pos, axis in enumerate(universe):
            if axis not in axes:
                array = np.expand_dims(array, axis=pos)
        return array

    letters = [''.join(chr(97 + universe.index(a)) for a in axes)
               for _c, _e, _m, axes in data]
    expr = (','.join(letters) + '->'
            + ''.join(chr(97 + universe.index(a)) for a in out))
    value = np.einsum(expr, *[d[0] for d in data], optimize=True)

    full_shape = [package.lengths[a] for a in universe]
    total_mask = np.ones(full_shape, dtype=bool)
    worst = np.ones(full_shape)
    base = np.ones(full_shape)
    for c, e, m, axes in data:
        # 显式按逻辑轴名对齐到 universe，杜绝物理轴序错位。
        c_full = np.broadcast_to(to_universe(c, axes), full_shape)
        e_full = np.broadcast_to(to_universe(e, axes), full_shape)
        m_full = np.broadcast_to(to_universe(m, axes), full_shape)
        total_mask &= m_full
        worst = worst * (np.abs(c_full) + e_full)
        base = base * np.abs(c_full)
    bound = ((worst - base) * total_mask).sum(
        axis=tuple(range(len(out), len(universe))))
    return value, bound, out
