"""张量收缩执行器命令行入口。

示例
----
# 跑内置场景并打印可核对的交付账本
python3 -I run_executor.py samples/scene.json

# 换工作区额度
python3 -I run_executor.py samples/scene.json --budget 128

# 换资料送达顺序（timeline 为 JSON 数组文件）
python3 -I run_executor.py samples/scene.json --timeline my_order.json

# 换模型包（新 revision，旧数值中间量不复用）
python3 -I run_executor.py samples/scene.json --package samples/axis-revision.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensor_exec.engine import ExecutionEngine  # noqa: E402
from tensor_exec.package import ModelPackage  # noqa: E402


def run(scene_path: Path, budget: int | None,
        timeline_path: Path | None, package_path: Path | None) -> dict:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    package_raw = (
        json.loads(package_path.read_text(encoding="utf-8"))
        if package_path else scene
    )
    package = ModelPackage(package_raw)
    used_budget = int(budget if budget is not None
                      else scene.get("budget_bytes", 0))
    timeline = (
        json.loads(timeline_path.read_text(encoding="utf-8"))
        if timeline_path else scene["timeline"]
    )
    engine = ExecutionEngine(package, used_budget)
    engine.run_timeline(timeline)
    return engine.report()


def main() -> None:
    parser = argparse.ArgumentParser(description="低秩多头因子张量收缩执行器")
    parser.add_argument("scene", type=Path, help="场景交换文件")
    parser.add_argument("--budget", type=int, default=None,
                        help="覆盖工作区额度（字节）")
    parser.add_argument("--timeline", type=Path, default=None,
                        help="覆盖资料送达顺序的 JSON 时间线文件")
    parser.add_argument("--package", type=Path, default=None,
                        help="替换为另一个合法模型包（按其 revision 解释）")
    args = parser.parse_args()
    report = run(args.scene, args.budget, args.timeline, args.package)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
