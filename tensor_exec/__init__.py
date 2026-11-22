"""低秩多头因子张量收缩执行器。"""

from .blocks import Block
from .contraction import (
    HeadResult,
    SharedIntermediate,
    apply_private,
    build_shared,
    execute_head,
)
from .engine import Delivery, ExecutionEngine
from .intervals import OutputAccumulator, term_product
from .package import FactorModel, HeadSpec, ModelPackage, Slice
from .runner import run_scene

__all__ = [
    "Block",
    "HeadResult",
    "SharedIntermediate",
    "apply_private",
    "build_shared",
    "execute_head",
    "Delivery",
    "ExecutionEngine",
    "OutputAccumulator",
    "term_product",
    "FactorModel",
    "HeadSpec",
    "ModelPackage",
    "Slice",
    "run_scene",
]
