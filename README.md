# 低秩多头因子收缩

samples/scene.json 是本地合成的命名轴、因子块和消费窗口。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 -m unittest discover -s tests -v
```

## 张量收缩执行器

`tensor_exec/` 实现从因子分片到可送下一层结果的块级流式收缩执行器，
`run_executor.py` 是命令行入口，`tests/` 含与暴力 `einsum` 参考实现
（`tests/reference.py`）的交叉验证。

```sh
# 跑场景并打印交付账本（原因、版本、误差界、工作区、共享占用）
python3 -I run_executor.py samples/scene.json
# 换工作区额度（超限只挂起，不会超额交付）
python3 -I run_executor.py samples/scene.json --budget 128
# 换资料送达顺序
python3 -I run_executor.py samples/scene.json --timeline my_order.json
# 换模型包：按新 revision 解释，旧数值中间量不复用
python3 -I run_executor.py samples/scene.json --package samples/axis-revision.json
```

### 模块划分

| 模块 | 职责 |
| --- | --- |
| `tensor_exec/blocks.py` | 因子块：逻辑轴、物理切块、结构零、原值/压缩双视图与解码 |
| `tensor_exec/package.py` | 模型包：命名轴网格、结构零与缺块区分、视图到达状态、头契约 |
| `tensor_exec/intervals.py` | 公开误差契约：含二阶项的区间传播与 FP64/FP32 舍入界 |
| `tensor_exec/contraction.py` | 块级收缩、按名对齐、子区域裁剪、公共子表达式与共享物化 |
| `tensor_exec/engine.py` | 时间线调度、额度裁决、报警原子交付/趋势局部交付、审计账本 |
| `tensor_exec/runner.py` | 场景装载与可重复重跑 |

### 关键语义

- **按逻辑轴名对齐**：因子物理轴序可在版本间调整（如 p8 的 frequency），
  收缩前按名转置；两个同形因子不会因同形而换轴。
- **结构零 vs 缺块**：结构零块跳过实际乘加且误差贡献为零；缺块使对应
  输出坐标不可证，绝不以零冒充。
- **误差契约**：每坐标维护 `ΣΠ|c|` 与 `ΣΠ(|c|+e)`，界为二者之差加累积
  舍入；符号相消导致输出很小时仍按界裁决，不用输出值做唯一判断。
- **报警原子性**：同组切片覆盖与误差全部成立才整体交付，等候中仍计工作区。
- **趋势局部性**：合格切片独立交付，账本记录所用包修订与实际误差界。
- **共享可行性**：公共因子的可提前消去轴先收缩成一份中间张量，缓存命中
  时跳过重复乘加与读取；临时区超限或误差不达标则回退各自交付。
- **版本隔离**：新模型包、新输出请求或视图级别变化只作废数值中间量，
  满足公开结构条件的执行决策可复用。

