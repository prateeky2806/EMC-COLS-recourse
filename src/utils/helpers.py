"""
This module containts helper functions to load data_loaders and get meta deta.
"""

import os

import torch
import pandas as pd

from loaders.data_loaders.pandas_interface import PandasDataLoader
from loaders.model_loaders.pytorch_interface import PyTorchModel
from explainers.dice import DicePyTorchExplainer
from explainers.search_methods import RandomSearch, ParallelLocalSearch, LocalSearch, RandomSparseSearch
import explainers.lime_tabular as lime_tabular
from train_model import PL_Trainer
from utils.data_io import DataIO


def get_lime_queries_from_dict(query_dict, pandas_loader):
    query_df = pd.DataFrame([query_dict], columns=query_dict.keys())
    return pandas_loader.encode_catdf_to_intdf(query_df).values.astype(float)

def get_lime_data(pandas_data_obj, split='train'):
    if split == 'train':
        data = pandas_data_obj.train_df.copy().drop(pandas_data_obj.outcome_name, 1)
    elif split == 'val':
        data = pandas_data_obj.val_df.copy().drop(pandas_data_obj.outcome_name, 1)
    elif split == 'test':
        data = pandas_data_obj.test_df.copy().drop(pandas_data_obj.outcome_name, 1)

    return pandas_data_obj.encode_catdf_to_intdf(data).values.astype(float)

def get_dice_data(pandas_dataloader):
    return pandas_dataloader.test_df.copy().drop(pandas_dataloader.outcome_name, 1).to_dict(orient='records')

def normalise_weights(val, minx, maxx):
    return ((val - minx)/(maxx - minx))*2

def setup_lime(pandas_data_obj):
    lime_train_data = get_lime_data(pandas_data_obj, 'train')
    lime_test_data = get_lime_data(pandas_data_obj, 'test')
    explainer = lime_tabular.LimeTabularExplainerPD(lime_train_data,
                                                    feature_names=pandas_data_obj.feature_names,
                                                    class_names=None,
                                                    categorical_features=pandas_data_obj.categorical_feature_indexes,
                                                    categorical_names=pandas_data_obj.categorical_map_int2liststr,
                                                    kernel_width=3,
                                                    feature_selection='none')
    return explainer, lime_train_data, lime_test_data

def setup_dice(pandas_data_obj, model):
    queries = get_dice_data(pandas_data_obj)
    exp = DicePyTorchExplainer(pandas_data_obj, model)
    return exp, queries

def setup_exp(pandas_data_obj, model, evaluator, type):
    # queries = get_dice_data(pandas_data_obj)
    if type == 'dice' or type == 'lime_dice':
        exp = DicePyTorchExplainer(pandas_data_obj, model)
    elif type == 'random':
        exp = RandomSearch(pandas_data_obj, model, evaluator)
    elif type == 'random_sparse':
        exp = RandomSparseSearch(pandas_data_obj, model, evaluator)
    elif type == 'exhaustive':
        exp = ExhaustiveSearch(pandas_data_obj, model, evaluator)
    elif type == 'pls':
            exp = ParallelLocalSearch(pandas_data_obj, model, evaluator)
    elif type == 'ls':
            exp = LocalSearch(pandas_data_obj, model, evaluator)
    return exp# , queries

def setup_pandas(io_obj):
    pandas_loader_params = {}
    pandas_loader_params['io_obj'] = io_obj
    pandas_loader_params['dataframe'] = io_obj.data
    pandas_loader_params['continuous_features'] = io_obj.dataset.ordinal_fnames
    pandas_loader_params['outcome_name'] = io_obj.dataset.class_target
    pandas_data_obj = PandasDataLoader(pandas_loader_params)
    return pandas_data_obj

def setup_data(args, balance=True):
    io_obj = DataIO(args)
    pandas_data_obj = setup_pandas(io_obj)
    return pandas_data_obj

def setup_model(args):
    ML_modelpath = os.path.join(args.project_dir, 'trained_models', f'{args.data_name}-default.ckpt')
    print(f'Model loaded from: {ML_modelpath}')
    pl_model = PL_Trainer.load_from_checkpoint(ML_modelpath)
    model = PyTorchModel(model=pl_model, backend=args.backend)
    predict_fn = lambda x: model.get_pred_probs(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    return model, predict_fn


class DDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

