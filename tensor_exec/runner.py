"""场景运行器：从交换文件驱动一次完整的时间线执行。

调用者可以：

* 换机器：载入不同场景文件或不同模型包（revision 变化时数值缓存失效）；
* 换轴切片：构造自定义 ``HeadSpec``/``Slice`` 后直接用
  :class:`~tensor_exec.engine.ExecutionEngine` 重跑；
* 换资料送达顺序：对 ``timeline`` 重排后重跑，交付结果只取决于已到达
  的资料状态，与顺序无关，账本完整记录每次决定的原因和版本。
"""

from __future__ import annotations

import json
from pathlib import Path

from .engine import ExecutionEngine
from .package import ModelPackage


def load_scene(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_scene(path: str | Path, budget_bytes: int | None = None,
              timeline_override: list[list[str]] | None = None) -> dict:
    """载入场景并跑完整个时间线，返回审计报告。"""
    raw = load_scene(path)
    package = ModelPackage(raw)
    budget = int(budget_bytes if budget_bytes is not None
                 else raw.get("budget_bytes", 0))
    timeline = timeline_override if timeline_override is not None else raw["timeline"]
    engine = ExecutionEngine(package, budget)
    engine.run_timeline(timeline)
    return engine.report()


def rerun_with_package(scene_path: str | Path, new_package_path: str | Path,
                       timeline: list[list[str]] | None = None) -> dict:
    """用新的合法模型包重跑同一时间线（旧数值中间量不可复用）。"""
    scene = load_scene(scene_path)
    new_raw = load_scene(new_package_path)
    package = ModelPackage(new_raw)
    engine = ExecutionEngine(package, int(scene["budget_bytes"]))
    engine.run_timeline(timeline if timeline is not None else scene["timeline"])
    return engine.report()
