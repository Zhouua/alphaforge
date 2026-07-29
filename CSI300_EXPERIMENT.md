# AlphaForge 与 AlphaMining 对齐实验

本仓库的对齐协议只有一个真源：
[`experiment_protocol.py`](experiment_protocol.py)。便于人工审计的等价 JSON
位于 [`config/alphamining_aligned.json`](config/alphamining_aligned.json)。

## 固定协议

- 数据：Qlib `cn_data`；字段为 `open/high/low/close/volume/vwap`，直接使用
  provider 原生字段（`raw=False`），不额外乘除 `$factor`。
- 股票池：CSI300；benchmark 为 `SH000300`。
- Train：2010-01-01 至 2019-11-30。
- Validation：2020-01-01 至 2021-11-30。
- Test：2022-01-01 至 2025-12-31。
- 标签：`Ref($open, -11) / Ref($open, -1) - 1`。信号日为 t，在
  open[t+1] 买入、open[t+11] 卖出，持有 10 个交易日。
- 模型：Qlib `LinearModel(estimator="ridge", alpha=10.0,
  fit_intercept=False, include_valid=False)`。
- 策略：`TopkDropoutStrategy(topk=50, n_drop=5)`，开盘成交。
- 初始资金 1 亿元；买入 5 bps，卖出 15 bps；单笔最低费用 5 元。
- 没有额外指定涨跌停过滤，因此公共评估明确使用
  `limit_threshold=None`。若 AlphaMining 使用 9.5% 限制，两边必须一起改。

Qlib 的 TopK-Drop 中 `n_drop=5` 表示每日最多替换 50 只持仓中的 5 只，即
单边替换比例不超过 10%；若按买卖双边成交额口径统计，数值约为 20%。

## 数据隔离

流程被强制拆成三个阶段：

1. `train_AFF.py` 只允许 train/validation 日期，任何 test 参数都会报错；
   因子搜索和候选排序只读取 train。
2. `freeze_factor_model.py` 只加载 train/validation，用 train 拟合固定 Ridge，
   在 validation 上报告结果，然后保存系数与因子顺序。
3. `public_test.py` 是唯一读取 test 的入口。它只接受冻结模型，不重选因子、
   不改方向、不重新拟合、不调超参。

test 末端标签需要 2025-12-31 之后至少 11 个交易日的 open 数据。公共评估会
核对完整测试日历，数据不足时直接失败，不会悄悄缩短测试期。

## 服务器环境（无 Docker）

不要安装 Docker，也不要使用 Docker-in-Docker。已有 pyenv 3.11.8 时可运行：

```bash
bash scripts/bootstrap_pyenv.sh
source .venv311/bin/activate
```

脚本只创建 Python venv 并安装
[`requirements-server.txt`](requirements-server.txt)。若平台镜像已经提供
PyTorch/Qlib，可直接激活原环境后从下面的数据检查开始。

设置 Qlib 数据路径：

```bash
export ALPHAFORGE_QLIB_PATH=/path/to/.qlib/qlib_data/cn_data
```

建议先确认 train/validation 范围的数据字段、动态成分股和 benchmark：

```bash
python validate_qlib_data.py --qlib_path="$ALPHAFORGE_QLIB_PATH"
```

该预检明确不读取 test；test 日历和标签右边界由公共评估入口自行检查。

## Stage 1：搜索并导出因子库

单个 seed：

```bash
python train_AFF.py \
  --instruments=csi300 \
  --train_start=2010-01-01 --train_end=2019-11-30 \
  --valid_start=2020-01-01 --valid_end=2021-11-30 \
  --qlib_path="$ALPHAFORGE_QLIB_PATH" \
  --seeds='[0]' --save_name=alphamining_aligned \
  --zoo_size=100 \
  --initial_candidates=10000 \
  --candidates_per_round=1000 \
  --max_rounds=15 \
  --device=auto
```

建议正式实验使用 `--seeds='[0,1,2,3,4]'`。每个 seed 的目录名包含完整的
train/validation 边界，主要产物为：

- `factor_library.json`：可移植公式库、train score 和协议元数据；
- `csv_zoo_final.csv`：便于人工浏览；
- `z_bld_zoo_final.pkl`：保留 AlphaForge 原生对象；
- `run_metadata.json`：候选数、轮次、设备和隔离声明。

`combine_AFF.py` 是原仓库的动态 OLS 组合器，不属于本次 Ridge 对齐协议，
因此不要用它生成主结果。

## Stage 2：用 train/validation 冻结共同模型

```bash
python freeze_factor_model.py \
  --factor_library=out/<RUN>/factor_library.json \
  --qlib_path="$ALPHAFORGE_QLIB_PATH" \
  --output_dir=frozen_models/alphaforge_seed0 \
  --max_factors=100 \
  --device=auto
```

冻结产物 `frozen_model.json` 包含完整因子顺序、Ridge 系数、intercept、
validation 指标以及 `created_without_test_data=true`。公共评估会验证这些字段。

## Stage 3：公共 test 评估

只有公共评估方运行：

```bash
python public_test.py \
  --frozen_model=frozen_models/alphaforge_seed0/frozen_model.json \
  --qlib_path="$ALPHAFORGE_QLIB_PATH" \
  --output_dir=public_test_results/alphaforge_seed0 \
  --device=auto
```

输出包括测试 IC/RankIC、逐日 IC、预测分数、持仓对象、组合日收益和扣费后风险
指标。策略成交价明确为 open。

## 接入 AlphaMining 并直观对比

如果 AlphaMining 已导出由 train 选出的公式 JSON 列表、`{"exprs": [...]}`、
`{"factors": [...]}` 或逐行文本，先包成同一 schema：

```bash
python wrap_factor_library.py \
  --input_path=/path/to/alphamining_factors.json \
  --method=AlphaMining \
  --output_path=factor_libraries/alphamining_seed0.json
```

公式必须能被 AlphaForge 的表达式解析器执行；如果 AlphaMining 使用另一套
公式语法，需要先做语法映射。之后对它运行完全相同的
`freeze_factor_model.py` 和 `public_test.py`。

复制 [`public_results_manifest.example.json`](public_results_manifest.example.json)
并填写各方法的 `summary.json`，生成同列比较表：

```bash
python compare_public_results.py \
  --manifest=public_results_manifest.json \
  --output=public_test_comparison.csv
```

至少报告 5 个 seed 的均值与标准差，同时保留 unique expressions evaluated、
wall-clock、有效因子数和换手数据。不得根据 test 结果选择 seed、因子数、方向、
阈值或任何超参数。
