# AlphaForge(AFF)

For the Docker-free, AlphaMining-aligned CSI300 protocol with an O2O 10-day
target, Qlib Ridge, leakage-safe factor export, and public TopK-Drop test, see
[`CSI300_EXPERIMENT.md`](CSI300_EXPERIMENT.md).


### Data Preparation
Similar to [AlphaGen](https://github.com/RL-MLDM/alphagen), We Use [Qlib](https://github.com/microsoft/qlib#data-preparation) as data save tool and download data from free & open-source data source  [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5).

Please install Qlib [Qlib](https://github.com/microsoft/qlib) first

Then download stock data through running `data_collection/fetch_baostock_data.py`

Then pass the downloaded provider with `--qlib_path=/path/for/qlib_data`,
or set `ALPHAFORGE_QLIB_PATH`. The fallback is
`~/.qlib/qlib_data/cn_data`.


### Run Our Model

#### stage1: Minning alpha factors
```shell
python train_AFF.py \
  --instruments=csi300 \
  --train_start=2010-01-01 --train_end=2019-11-30 \
  --valid_start=2020-01-01 --valid_end=2021-11-30 \
  --seeds='[0,1,2,3,4]' --save_name=experiment --zoo_size=100
```

Here,
- `instruments` is the dataset to use, e.g., `csi300`,`csi500`.
- `seeds` is random seed list, e.g., `[0,1,2]` or `[0]`. 
- `save_name` is the prefix when saving results. `zoo_size` is the number of
  factors exported to `factor_library.json`.
- Stage 1 rejects test dates by design.

#### stage2: Combining alpha factors (legacy)

`combine_AFF.py` reproduces the original dynamic combiner. It is not part of
the AlphaMining-aligned Ridge protocol; use `freeze_factor_model.py` and
`public_test.py` from [`CSI300_EXPERIMENT.md`](CSI300_EXPERIMENT.md) for the
main comparison.
```shell
python combine_AFF.py --instruments=csi300 --train_end_year=2020 --seeds=[0,1,2,3,4] --save_name=test --n_factors=10 --window=inf
```
Here `instruments,train_end_year,seeds,save_name`,` must be the same as it in stage 1
- `n_factors` is the num of factors used at each day, it should be less than or equal to `zoo_size` in stage 1.
- `window` is the slicing window that is used to evaluate the alpha factors in order to dynamicly select and cobine.

#### stage3: Show the results

You could run the ipython notebook file 

```shell
exp_AFF_calc_result.ipynb
```

to generate and concat experiment result.


### Run baseline experiments

The experiment process of other models is similar to running our AFF model, Except that none of the other models have a combine step.

#### GP:

train: `train_GP.py`, show result: `exp_GP_calc_result.ipynb`

#### RL:

train: `train_RL.py`, show result: `exp_RL_calc_result.ipynb`

#### DSO:

train: `train_DSO.py`, show result: `exp_DSO_calc_result.ipynb`

#### ML models including XGBoost, LightGBM and MLP:

train & show results: `exp_ML_train_and_result.ipynb`
