
import os
import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--instrument',type=str,default='csi300')
parser.add_argument('--seed',type=str,default='[0,1,2,3,4]')
parser.add_argument('--years',type=str,default=None)
parser.add_argument('--freq',type=str,default='day')
parser.add_argument('--cuda',type=str,default='0')
parser.add_argument('--device',type=str,default='auto')
parser.add_argument('--train_start',type=str,default='2010-01-01')
parser.add_argument('--train_end',type=str,default='2019-11-30')
parser.add_argument('--valid_start',type=str,default='2020-01-01')
parser.add_argument('--valid_end',type=str,default='2021-11-30')
parser.add_argument('--test_start',type=str,default='2022-01-01')
parser.add_argument('--test_end',type=str,default='2025-12-31')
parser.add_argument('--qlib_path',type=str,default=None)
parser.add_argument('--population_size',type=int,default=1000)
parser.add_argument('--generations',type=int,default=40)


args = parser.parse_args()
instruments = args.instrument
args.seed = eval(args.seed)
if args.years is not None:
    args.years = eval(args.years)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
if args.device == "auto":
    if torch.backends.mps.is_available():
        args.device = "mps"
    elif torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
print('instruments',instruments)
print('seed',args.seed)
print('years',args.years)
print('cuda',args.cuda)
print('device',args.device)


import json
from collections import Counter

import numpy as np

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from gan.utils.data import get_data_by_year


def mean_rank_ic(factor, label, chunk_size=64):
    values = []
    for start in range(0, len(factor), chunk_size):
        values.append(
            batch_spearmanr(
                factor[start:start + chunk_size],
                label[start:start + chunk_size],
            )
        )
    return torch.cat(values).mean().item()


def _metric(x, y, w):
    key = y[0]

    if key in cache:
        return cache[key]
    token_len = key.count('(') + key.count(')')
    if token_len > 20:
        return -1.

    expr = eval(key)
    try:
        factor = expr.evaluate(data)
        factor = normalize_by_day(factor)
        ic = batch_pearsonr(factor, target_factor)
        ic = abs(torch.nan_to_num(ic).mean().item())
    except OutOfDataRangeError:
        ic = -1.
    if np.isnan(ic):
        ic = -1.
    cache[key] = ic
    return ic




def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    try:
        v_valid = eval(top_key).evaluate(data_valid)
        ic_valid = batch_pearsonr(v_valid, target_factor_valid)
        ic_valid = torch.nan_to_num(ic_valid,nan=0,posinf=0,neginf=0).mean().item()
        ric_valid = mean_rank_ic(v_valid, target_factor_valid)
        return {'ic_valid': ic_valid, 'ric_valid': ric_valid}
    except OutOfDataRangeError:
        print ('Out of data range')
        print(top_key)
        return {'ic_valid': -1., 'ric_valid': -1.}


def build_pool(capacity):
    pool = AlphaPool(capacity=capacity,
                    stock_data=data,
                    target=target,
                    ic_lower_bound=None)

    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    pool.force_load_exprs(exprs)
    return pool


def try_pool(capacity):
    pool = build_pool(capacity)
    ic_valid, ric_valid = pool.test_ensemble(data_valid, target)
    return {'ic_valid': ic_valid, 'ric_valid': ric_valid}




def ev():
    global generation
    generation += 1
    res = (
        [{'pool': 0, 'res': try_single()}] +
        [
            {'pool': cap, 'res': try_pool(cap)}
            for cap in (10, 20, 50, 100)
        ]
    )
    valid_only = [
        {
            'pool': item['pool'],
            'valid_ic': item['res']['ic_valid'],
            'valid_rank_ic': item['res']['ric_valid'],
        }
        for item in res
    ]
    print(valid_only)
    global save_dir
    dir_ = save_dir
    #'/path/to/save/results'
    os.makedirs(dir_, exist_ok=True)
    if generation % 2 == 0:
        with open(f'{dir_}/{generation}.json', 'w') as f:
            json.dump({'cache': cache, 'valid': valid_only}, f)





explicit_dates = [
    args.train_start,
    args.train_end,
    args.valid_start,
    args.valid_end,
    args.test_start,
    args.test_end,
]
if args.years is None:
    if not all(value is not None for value in explicit_dates):
        parser.error("Pass all six explicit split dates.")
    runs = [None]
else:
    runs = args.years

for seed in args.seed:
    for train_end_year in runs:
        #'/path/to/save/results'
        if train_end_year is None:
            split_id = (
                f"{args.train_start}_{args.train_end}_{args.valid_start}_"
                f"{args.valid_end}_{args.test_start}_{args.test_end}"
            )
        else:
            split_id = str(train_end_year)
        save_dir = f'out_gp/{instruments}_{split_id}_{args.freq}_{seed}'

        Metric = make_fitness(function=_metric, greater_is_better=True)
        funcs = [make_function(**func._asdict()) for func in generic_funcs]

        generation = 0
        cache = {}

        reseed_everything(seed)


        if train_end_year is None:
            from gan.utils.data import get_data_by_dates
            returned = get_data_by_dates(
                train_start=args.train_start,
                train_end=args.train_end,
                valid_start=args.valid_start,
                valid_end=args.valid_end,
                test_start=args.test_start,
                test_end=args.test_end,
                instruments=instruments,
                target=target,
                freq=args.freq,
                qlib_path=args.qlib_path,
                device=args.device,
            )
        else:
            returned = get_data_by_year(
                train_start=2010,
                train_end=train_end_year,
                valid_year=train_end_year + 1,
                test_year=train_end_year + 2,
                instruments=instruments,
                target=target,
                freq=args.freq,
                qlib_path=args.qlib_path,
                device=args.device,
            )
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned

        pool = AlphaPool(capacity=10,
                        stock_data=data,
                        target=target,
                        ic_lower_bound=None)

        target_factor = target.evaluate(data)
        target_factor_valid = target.evaluate(data_valid)
        target_factor_test = target.evaluate(data_test)

        
        features = ['open_', 'close', 'high', 'low', 'volume', 'vwap']
        constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
        terminals = features + constants

        X_train = np.array([terminals])
        y_train = np.array([[1]])

        est_gp = SymbolicRegressor(population_size=args.population_size,
                                generations=args.generations,
                                init_depth=(2, 6),
                                tournament_size=600,
                                stopping_criteria=1.,
                                p_crossover=0.3,
                                p_subtree_mutation=0.1,
                                p_hoist_mutation=0.01,
                                p_point_mutation=0.1,
                                p_point_replace=0.6,
                                max_samples=0.9,
                                verbose=1,
                                parsimony_coefficient=0.,
                                random_state=seed,
                                function_set=funcs,
                                metric=Metric,
                                const_range=None,
                                n_jobs=1)
        est_gp.fit(X_train, y_train, callback=ev)
        valid_candidates = (
            [{'pool': 0, 'res': try_single()}] +
            [
                {'pool': cap, 'res': try_pool(cap)}
                for cap in (10, 20, 50, 100)
            ]
        )
        selected = max(
            valid_candidates,
            key=lambda item: abs(item['res']['ic_valid']),
        )
        selected_capacity = selected['pool']
        orientation = 1. if selected['res']['ic_valid'] >= 0 else -1.
        if selected_capacity == 0:
            selected_expr = Counter(cache).most_common(1)[0][0]
            valid_prediction = normalize_by_day(
                eval(selected_expr).evaluate(data_valid)
            )
            test_prediction = normalize_by_day(
                eval(selected_expr).evaluate(data_test)
            )
            selected_pool = {
                'exprs': [selected_expr],
                'weights': [orientation],
            }
        else:
            selected_alpha_pool = build_pool(selected_capacity)
            valid_prediction = selected_alpha_pool.predict_ensemble(data_valid)
            test_prediction = selected_alpha_pool.predict_ensemble(data_test)
            selected_pool = selected_alpha_pool.to_dict()
            selected_pool['weights'] = [
                orientation * weight for weight in selected_pool['weights']
            ]
        valid_prediction = valid_prediction * orientation
        test_prediction = test_prediction * orientation
        valid_ic = batch_pearsonr(
            valid_prediction, target_factor_valid
        ).mean().item()
        valid_rank_ic = mean_rank_ic(valid_prediction, target_factor_valid)
        test_ic = batch_pearsonr(
            test_prediction, target_factor_test
        ).mean().item()
        test_rank_ic = mean_rank_ic(test_prediction, target_factor_test)
        library_capacity = min(100, len(cache))
        candidate_library = build_pool(library_capacity).to_dict()
        final_results = {
            'selection_metric': 'absolute validation IC',
            'selected_capacity': selected_capacity,
            'pool': selected_pool,
            'candidate_library': candidate_library,
            'valid_ic': valid_ic,
            'valid_rank_ic': valid_rank_ic,
            'test_ic': test_ic,
            'test_rank_ic': test_rank_ic,
            'unique_expressions_evaluated': len(cache),
            'valid_candidates': valid_candidates,
        }
        torch.save(
            valid_prediction.detach().cpu(),
            f'{save_dir}/pred_valid.pt',
        )
        torch.save(
            test_prediction.detach().cpu(),
            f'{save_dir}/pred_test.pt',
        )
        with open(f'{save_dir}/final.json', 'w') as f:
            json.dump({'cache': cache, 'result': final_results}, f, indent=2)
        print(final_results)
