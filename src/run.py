"""
This script can run both Lime and Dice from a trained model by pytorch lightening.
"""

import os, argparse, timeit
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm
import random
import copy
from pathlib import Path
from itertools import product

from utils.metrics import MetricsEvaluator, UserEvaluator
from utils.helpers import *
from train_model import prepare_data_for_training


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Data IO params
    parser.add_argument('--data_name', default='adult_binary', type=str)
    parser.add_argument('--project_dir', default='../', type=str)
    parser.add_argument('--data_folder', default='data', type=str, help='Name of the folder containing data inside project Dir.')
    # parser.add_argument('--max_test_samples', default=500, type=int, help='maximum number of test samples')
    parser.add_argument('--save_data', action='store_true', help='Save the data splits.')
    parser.add_argument('--dump_negative_data', action='store_true', help='Test the test data and dump only negative class in test.')
    # Cost sampling
    parser.add_argument('--num_mcmc', default=1000, type=int, help='number of mcmc sample.')

    parser.add_argument('--alpha', default=None, type=float, help='multiplicative factor of linear cost, 0=linear, 1=percentile, frac-convex combination.')
    parser.add_argument('--test_alpha', default=None, type=float, help='multiplicative factor of linear cost, 0=linear, 1=percentile, frac-convex combination.')
    parser.add_argument('--load_type', default='', type=str)
    # parser.add_argument('--num_mcmc', default=5, type=int)
    # Explainer params
    parser.add_argument('--model', default='ls', type=str)
    parser.add_argument('--eval', default='cost', type=str, choices=['diversity', 'proximity', 'sparsity',
                                                                          'cost_simple', 'cost', 'dirichlet'])
    parser.add_argument('--thresh', nargs="+", default=[1, 2, 3])
    parser.add_argument('--score_type', default='min', type=str)
    parser.add_argument('--desired_class', default='opposite', type=str)
    parser.add_argument('--features_to_vary', default='all', type=str)
    parser.add_argument('--permitted_range', default=None, type=str)
    # Experiment params
    parser.add_argument('--run_id', default=0, type=int, help='Id of the runs.')

    parser.add_argument('--num_users', default=None, type=int, help='number of counterfactual required.')
    parser.add_argument('--num_cfs', default=10, type=int, help='number of counterfactual required.')
    parser.add_argument('--backend', default='PYT', type=str)
    parser.add_argument('--verbose', action='store_true')
    # Post Hoc Sparsity
    parser.add_argument('--posthoc_sparsity_param', default=0.2, type=int, help='')
    parser.add_argument('--posthoc_sparsity_algorithm', default='linear', type=str, help='')
    # Search related params
    parser.add_argument('--budget', default=1000, type=int, help='Number of forward passes.')
    parser.add_argument('--num_parallel_runs',  default=5, type=int, help='PLS: number of parallel runs')
    parser.add_argument('--hamming_dist',  default=2, type=int, help='LS: Hamming distance type used. # of features to edit.')
    parser.add_argument('--perturb_type',  default='all', type=str, help='LS: number of elements to perturb in the set.')
    parser.add_argument('--init_type',  default='ham_2', type=str, help='LS: Initial initialization method for local search. [random, ham_x, org]')
    parser.add_argument('--iter_type',  default='best', type=str, help='LS: at each step used the last step samples or the best ones. [best, linear]')

    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--run_name', default='exp', type=str)
    # General params
    parser.add_argument('--seed', default=-1, type=int, help='seed for np and random.')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()


    if args.seed == -1:
        print('RANDOM SEED')
        args.seed = np.random.randint(1, 1000)

    np.random.seed(args.seed)
    random.seed(args.seed)
    start_time = timeit.default_timer()

    current_time = datetime.now().strftime('%b%d_%H-%M-%s')
    exp_name = args.project_name
    args.run_name = f'{args.run_name}_{current_time}'

    # Loading Data, model, explainers
    pandas_data_obj = setup_data(args)
    users_info = pandas_data_obj.io_obj.dataset.users_info
    users, cost_map = users_info['users'], users_info['cost_map']
    if args.num_users == None:
        args.num_users = len(users)
    metric_evaluator = MetricsEvaluator(pandas_data_obj)
    cost_evaluator = UserEvaluator(pandas_data_obj, users_info, score_type=args.score_type)
    model, predict_fn = setup_model(args)

    x = prepare_data_for_training(pandas_data_obj, split='test')[0]
    preds = np.argmax(
        model.predict_probs(prepare_data_for_training(pandas_data_obj, split='test')[0], return_last=False),
        axis=1)
    assert preds.sum() == 0, 'all test samples need to have class 0 label under the model.'
    model.num_forward_pass = 0
    explainer, queries_lime, outputs = DDict(), DDict(), DDict()

    if 'lime' in args.model:
        explainer_lime, queries_lime['train'], queries_lime['test'] = setup_lime(pandas_data_obj)
        queries_lime['test'] = queries_lime['test'][:args.num_users]
        lime_weights = []

    if args.model != 'lime':
        if 'cost' in args.eval:
            explainer = setup_exp(pandas_data_obj, model, cost_evaluator, type=args.model)
        else:
            explainer = setup_exp(pandas_data_obj, model, metric_evaluator, type=args.model)
        queries_test = [user['user_cost']['query'] for user in users]

    cf_objs = []
    print(f'Running iteration of {args.model} model')
    for user_id in tqdm(range(args.num_users)):
        if 'lime' in args.model:
            explainer.model.num_forward_pass = 0
            lw = explainer_lime.explain_instance(queries_lime['test'][user_id], predict_fn, pandas_loader=pandas_data_obj,
                                         num_features=queries_lime['test'].shape[1],
                                        num_samples=max(int(args.budget*0.1), 300)).as_list()
            feat, val = zip(*lw)
            if args.verbose:
                print(lw)
            fweights = {e[0]:e[1] for e in lw}
            w_min, w_max = min(list(fweights.values())), max(list(fweights.values()))
            f_norm = {k: normalise_weights(v, w_min, w_max) for k,v in fweights.items()}
            f_org = {feat[i]: val[i] for i in range(len(feat))}
            lime_weights.append({'original':f_org, 'normalised':f_norm})

        if args.model == 'lime_dice':
            dice_exp = explainer.generate_counterfactuals(queries_test[user_id],
                                                    total_CFs=args.num_cfs,
                                                    budget=args.budget,
                                                    feature_weights=lime_weights[user_id]['normalised'],
                                                    desired_class="opposite",
                                                    verbose=args.verbose)
            cf_objs.append(dice_exp)

        elif args.model == 'dice':
            explainer.model.num_forward_pass = 0
            dice_exp = explainer.generate_counterfactuals(queries_test[user_id],
                                                    total_CFs=args.num_cfs,
                                                    budget=args.budget,
                                                    desired_class="opposite",
                                                    verbose=args.verbose,
                                                    seed=args.seed)
            cf_objs.append(dice_exp)

        elif args.model in ['random', 'random_sparse', 'exhaustive', 'pls', 'ls']:
            kwargs = vars(args)
            kwargs['user_id'] = user_id
            kwargs['eval'] = args.eval
            explainer.model.num_forward_pass = 0
            explainer.model.local_forward = 0
            random_exp = explainer.generate_counterfactuals(queries_test[user_id], **kwargs)
            cf_objs.append(random_exp)

    if args.model not in ['lime']:
        # Get data for metrics
        cfs_list = [copy.deepcopy(obj.final_cfs_list) for obj in cf_objs]
        num_for_pass = [obj.num_forward_pass for obj in cf_objs]
        original_class = [obj.test_pred for obj in cf_objs]
        metric_evaluator.get_metrics(copy.deepcopy(cfs_list), queries_test, original_class)
        qt_df = pd.DataFrame.from_records(queries_test)[pandas_data_obj.feature_names]
        cost_evaluator.get_metrics(qt_df, copy.deepcopy(cfs_list), original_class, args.thresh, args.test_alpha)
        dump_data = {'args': args, 'queries':queries_test, 'users': users, 'cost_map': cost_map,
                     'cfs_list': cfs_list, 'cfs_org_class': original_class, 'num_forward_pass': num_for_pass,

                     'metrics': metric_evaluator.metric_dict, 'metric_scores': metric_evaluator.scores,
                     'dirichlet_weights': metric_evaluator.dirichlet_weights,
                     'avg_metrics': metric_evaluator.avg_metric_dict, 'avg_metric_score': metric_evaluator.avg_score,

                     'costs': cost_evaluator.costs, 'user_score': cost_evaluator.user_score,
                     'pop_avg_score': cost_evaluator.pop_avg_score, 'invalid': cost_evaluator.invalid,
                     'pop_coverage': cost_evaluator.pop_coverage,
                     'pop_frac_satisfied': cost_evaluator.pop_frac_satisfied
                     }

        if args.model == 'lime_dice':
            dump_data['weights'] = lime_weights
            dump_data['queries_lime_train'] = queries_lime['train']
            dump_data['queries_lime_test'] = queries_lime['test']
    elif args.model == 'lime':
        dump_data = {'args':args, 'queries': queries_lime, 'weights': lime_weights}

    # Dumping data
    params = f'{exp_name}/{args.data_name}/{args.eval}'
    dump_dir = os.path.join(args.project_dir, 'saved_data', params, args.model)
    os.makedirs(dump_dir, exist_ok=True)
    if args.run_id != 0:
        fname = f'query_{args.num_users}_cfs_{args.num_cfs}_bud_{args.budget}_mcmc{args.num_mcmc}_alpha_{args.alpha}_runid{args.run_id}.pkl'
    else:
        fname = f'query_{args.num_users}_cfs_{args.num_cfs}_bud_{args.budget}_mcmc{args.num_mcmc}_alpha_{args.alpha}.pkl'

    dump_data['time'] = round((timeit.default_timer() - start_time)/60, 4)
    dump_data['path'] = os.path.join(dump_dir, fname)
    print(f'Total Time taken for {args.num_cfs} Counterfactuals is {dump_data["time"]:.4f} minutes')
    if not args.debug:

        print(f'Dumping data at {os.path.join(dump_dir, fname)}')
        pickle.dump(dump_data, open(os.path.join(dump_dir, fname), 'wb'))

    print('Done!')


