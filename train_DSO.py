import torch
import sklearn
import tensorflow as tf
import numpy as np
import os,json

from alphagen.data.expression import *
# from alphagen_qlib.calculator import QLibStockDataCalculator
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gan.utils.data import get_data_by_year

funcs = {func.name: Token(complexity=1, **func._asdict()) for func in generic_funcs}
for i, feature in enumerate(['open', 'close', 'high', 'low', 'volume', 'vwap']):
    funcs[f'x{i+1}'] = Token(name=feature, arity=0, complexity=1, function=None, input_var=i)
for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]:
    funcs[f'Constant({v})'] = HardCodedConstant(name=f'Constant({v})', value=v)

def main(
        instruments:str='csi300',
        train_end:int=None,
        train_start_date:str=None,
        train_end_date:str=None,
        valid_start:str=None,
        valid_end:str=None,
        test_start:str=None,
        test_end:str=None,
        qlib_path:str=None,
        seeds:list=[0],
        capacity:int=100,
        n_samples:int=5000,
        batch_size:int=128,
        cuda:int=0,
        name:str='test',
):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    if isinstance(seeds,str):
        seeds = eval(seeds)
    for seed in seeds:
        tf.set_random_seed(seed)
        reseed_everything(seed)
        explicit_dates = [
            train_start_date,
            train_end_date,
            valid_start,
            valid_end,
            test_start,
            test_end,
        ]
        if not any(value is not None for value in explicit_dates) and train_end is None:
            train_start_date, train_end_date = "2010-01-01", "2019-11-30"
            valid_start, valid_end = "2020-01-01", "2021-11-30"
            test_start, test_end = "2022-01-01", "2025-12-31"
            explicit_dates = [
                train_start_date,
                train_end_date,
                valid_start,
                valid_end,
                test_start,
                test_end,
            ]
        if any(value is not None for value in explicit_dates):
            if not all(value is not None for value in explicit_dates):
                raise ValueError("Pass all six explicit split dates or none of them.")
            from gan.utils.data import get_data_by_dates
            returned = get_data_by_dates(
                train_start=train_start_date,
                train_end=train_end_date,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
                instruments=instruments,
                target=target,
                freq='day',
                qlib_path=qlib_path,
            )
            split_id = (
                f"{train_start_date}_{train_end_date}_{valid_start}_{valid_end}_"
                f"{test_start}_{test_end}"
            )
        else:
            returned = get_data_by_year(
                train_start=2010,
                train_end=train_end,
                valid_year=train_end + 1,
                test_year=train_end + 2,
                instruments=instruments,
                target=target,
                freq='day',
                qlib_path=qlib_path,
            )
            split_id = str(train_end)
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned

        cache = {}

        X = np.array([['open_', 'close', 'high', 'low', 'volume', 'vwap']])
        y = np.array([[1]])
        functions.function_map = funcs

        pool = AlphaPool(capacity=capacity,
                        stock_data=data,
                        target=target,
                        ic_lower_bound=None)
        save_path = f'out_dso/{name}_{instruments}_{capacity}_{split_id}_{seed}/'
        os.makedirs(save_path,exist_ok=True)

        class Ev:
            def __init__(self, pool):
                self.cnt = 0
                self.pool = pool
                self.results = {}
                self.seen = set()

            def alpha_ev_fn(self, key):
                self.seen.add(key)
                expr = eval(key)
                try:
                    ret = self.pool.try_new_expr(expr)
                except OutOfDataRangeError:
                    ret = -1.
                self.cnt += 1
                if self.cnt % 100 == 0:
                    valid_ic = pool.test_ensemble(data_valid,target)[0]
                    self.results[self.cnt] = valid_ic
                    print(self.cnt, valid_ic)
                return ret

        ev = Ev(pool)

        config = dict(
            task=dict(
                task_type='regression',
                function_set=list(funcs.keys()),
                metric='alphagen',
                metric_params=[lambda key: ev.alpha_ev_fn(key)],
            ),
            training={
                'n_samples': n_samples,
                'batch_size': batch_size,
                'epsilon': 0.05,
            },
            prior={'length': {'min_': 2, 'max_': 20, 'on': True}},
            experiment={'seed':seed},
        )

        # Create the model
        model = DeepSymbolicRegressor(config=config)
        model.fit(X, y)
        valid_ic, valid_rank_ic = pool.test_ensemble(data_valid, target)
        orientation = 1. if valid_ic >= 0 else -1.
        test_ic, test_rank_ic = pool.test_ensemble(data_test, target)
        valid_prediction = pool.predict_ensemble(data_valid) * orientation
        test_prediction = pool.predict_ensemble(data_test) * orientation
        pool_result = pool.to_dict()
        pool_result['weights'] = [
            orientation * weight for weight in pool_result['weights']
        ]
        with open(f'{save_path}/pool.json', 'w') as f:
            json.dump(pool_result, f)
        torch.save(valid_prediction.detach().cpu(), f'{save_path}/pred_valid.pt')
        torch.save(test_prediction.detach().cpu(), f'{save_path}/pred_test.pt')
        metrics = {
            'orientation_selected_on_valid': orientation,
            'valid_ic': valid_ic * orientation,
            'valid_rank_ic': valid_rank_ic * orientation,
            'test_ic': test_ic * orientation,
            'test_rank_ic': test_rank_ic * orientation,
            'unique_expressions_evaluated': len(ev.seen),
            'valid_trace': ev.results,
        }
        with open(f'{save_path}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(metrics)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
