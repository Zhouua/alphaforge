# AlphaForge / GP / DSO 统一因子生成

## 统一实验边界

三种方法只负责提出公式并在训练集打分，使用完全相同的：

- Qlib `cn_data` 原生字段：`open/high/low/close/volume/vwap`
- 股票池：CSI300
- Train：2010-01-01 ～ 2019-11-30
- Validation：2020-01-01 ～ 2021-11-30
- Target：`Ref($open, -11) / Ref($open, -1) - 1`
- 最大表达式长度：20
- 特征历史热身：1000 个交易日（覆盖 20 层、单层最大 50 日回看）
- 因子库容量：100，硬性下限：50

生成进程没有 test 日期参数，也不会创建 test 数据缓存。生成后的三个
`factor_library.json` 再分别交给同一个 Qlib Ridge：

- estimator：Ridge
- alpha：10.0
- fit_intercept：False
- 只用 train 拟合，validation 报告并冻结

最后只有 `public_test.py` 可以读取 2022-01-01 ～ 2025-12-31。

## 三种生成原理

### AlphaForge

公式被编码成最长 20 的 token 序列。首先随机采样候选，并用训练集的日截面
IC、ICIR、覆盖率和唯一值比例做真实评价。预测器学习“表达式 token →
训练分数”，生成器再优化为更容易产生高预测分数、彼此有差异且处于有潜力
潜空间的表达式。新公式仍必须经过真实 Qlib 评价；达到 IC/ICIR 门槛且与已
入库因子的收益序列相关性不超过阈值后才进入 zoo。这个“预测器指导生成 →
真实评价 → 去重/去相关 → 回灌训练”的循环持续到 zoo 达到 100。

### GP

GP 维护一批表达式树。每一代用训练集绝对日截面 IC 作为适应度，通过锦标赛
选择父代，再执行子树交叉、子树变异、hoist 变异和节点变异。所有见过的唯一
合法公式进入训练分数缓存，搜索结束后按训练 IC 排序、规范化和去重，导出
前 100。validation 不参与 GP 适应度或库排序。

### DSO

DSO 用 LSTM 策略按前序遍历顺序逐 token 生成表达式树。每批公式得到训练集
绝对日截面 IC 奖励；risk-seeking policy gradient 只用奖励最高的
`epsilon=0.05` 部分更新策略，并用奖励分位数作为 baseline。长度、必须含
输入字段和均匀 arity prior 约束搜索空间。所有唯一合法公式进入训练分数
缓存，最后按训练 IC 导出前 100。这里使用官方 PyTorch DSO，不使用旧版
TensorFlow 1.14。

## 在 RD-Agent venv 中运行

```bash
source ~/work/RD-Agent/.venv/bin/activate
cd ~/work/alphaforge
export ALPHAFORGE_QLIB_PATH="$HOME/.qlib/qlib_data/cn_data"
```

AlphaForge：

```bash
bash scripts/run_final_factor_generation.sh
```

GP：

```bash
bash scripts/run_final_gp_generation.sh
```

DSO 首次只安装兼容性运行依赖，不安装旧 TensorFlow，也不运行官方
`setup.py`：

```bash
python -m pip install -r requirements-dso-pytorch.txt
bash scripts/run_final_dso_generation.sh
```

DSO 脚本会先用相同 Qlib 训练数据运行一个 32 表达式的完整 runtime
preflight，覆盖公式生成、AlphaForge 表达式解析、Qlib 奖励、PyTorch
策略更新和 DSO 日志收尾。只有出现 `PASS: DSO preflight` 后才会自动开始
20,000 样本正式搜索。诊断时可单独查看
`logs/dso_preflight_seed0.log`；正式实验不要设置
`DSO_SKIP_PREFLIGHT=1`。

默认输出：

```text
out/<run>/factor_library.json
out_gp/gp_csi300_<train>_<valid>_0/factor_library.json
out_dso/dso_csi300_<train>_<valid>_0/factor_library.json
```

每个目录同时包含可直接入库的 `factor_library_for_db.csv` 和证明
`test_data_loaded=false` 的 `run_metadata.json`。

## 统一 Ridge 冻结

以下 `<split>` 为
`2010-01-01_2019-11-30_2020-01-01_2021-11-30`：

```bash
python freeze_factor_model.py \
  --factor_library="out_gp/gp_csi300_<split>_0/factor_library.json" \
  --qlib_path="$ALPHAFORGE_QLIB_PATH" \
  --output_dir="frozen_models/gp_seed0" \
  --max_factors=100 \
  --device=cuda:0

python freeze_factor_model.py \
  --factor_library="out_dso/dso_csi300_<split>_0/factor_library.json" \
  --qlib_path="$ALPHAFORGE_QLIB_PATH" \
  --output_dir="frozen_models/dso_seed0" \
  --max_factors=100 \
  --device=cuda:0
```

AlphaForge 沿用相同命令和已经生成的因子库。先比较三个目录中的
`validation_metrics.json`，决定是否接受实验；一旦冻结，不再修改公式、
公式数、符号或 Ridge 参数。

## 公共 test 与最终对比

```bash
python public_test.py \
  --frozen_model=frozen_models/gp_seed0/frozen_model.json \
  --qlib_path="$ALPHAFORGE_QLIB_PATH" \
  --output_dir=public_test_results/gp_seed0 \
  --device=cuda:0

python public_test.py \
  --frozen_model=frozen_models/dso_seed0/frozen_model.json \
  --qlib_path="$ALPHAFORGE_QLIB_PATH" \
  --output_dir=public_test_results/dso_seed0 \
  --device=cuda:0
```

AlphaForge 也用同一个 `public_test.py`。把三个 `summary.json` 写入
`public_results_manifest.example.json` 后生成最终表：

```bash
python compare_public_results.py \
  --manifest=public_results_manifest.example.json \
  --output=public_test_comparison.csv
```
