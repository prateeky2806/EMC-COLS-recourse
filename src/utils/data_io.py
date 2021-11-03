"""bla"""
import os, copy
from tqdm import tqdm
import numpy as np
import pickle, argparse
import pandas as pd
from addict import Dict
from scipy.special import digamma, loggamma
from torch.distributions.kl import kl_divergence
from torch.distributions.dirichlet import Dirichlet
import torch
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))


'''
To match costs from various samples
for feat in user_cost['preference_scores']:
    print(user_cost['preference_scores'][feat])
    print(user_cost['means_costs'][feat])
    print(user_metric_costs[0]['preference_scores'][feat])
    print(user_metric_costs[0]['costs'][feat])

'''

class UserSampler:
    def __init__(self, pandas_obj, args, max_cost=99999, variance=0.001):
        from utils.helpers import get_dice_data

        self.dataclass = pandas_obj
        self.max_cost = max_cost
        self.variance = variance
        self.alpha = args.alpha
        self.pop_features = get_dice_data(self.dataclass)
        self.num_users = len(self.pop_features)
        self.set_types()
        _, self.original_ranges = self.dataclass.get_features_range()
        self.users, self.users_with_metric_info = self.sample_users(args.alpha)
        self.cost_map = {}
        for feat, cat in self.original_ranges.items():
            if feat in self.dataclass.continuous_feature_names:
                cat = self.cont_to_range(cat)
            self.cost_map[feat] = {c: ii for ii, c in enumerate(cat)}
        self.users_info = {'users': self.users, 'cost_map':self.cost_map}
        self.users_info_with_metric_info = {'users': self.users_with_metric_info, 'cost_map': self.cost_map}
        pass

    def set_types(self):
        self.od, self.uod, self.pod, self.fixed = [], [], [], []
        for feat, type in self.dataclass.feature_types.items():
            if type == 'ordered':
                self.od.append(feat)
            elif type == 'unordered':
                self.uod.append(feat)
            elif type == 'partial_ordered':
                self.pod.append(feat)
            elif type == 'fixed':
                self.fixed.append(feat)

    # @time_profile(sort_by='cumulative', lines_to_print=None, strip_dirs=True)
    def sample_users(self, alpha):
        users = []
        users_with_metric_info = []
        for user_id in tqdm(range(self.num_users)):
            # print(f'User {user_id}')
            if alpha != None:
                user_alpha = alpha
            else:
                user_alpha = np.round(np.random.uniform(0, 1), 2)
            user_cost = self.sample_single_user_costs(user_id, user_alpha)
            user_metric_costs = []
            # print(f'Num_mcmc: {self.dataclass.args.num_mcmc}')
            for i in range(self.dataclass.args.num_mcmc):
                user_metric_costs.append(self.sample_single_user_metric_costs(user_id, user_alpha))

            metric_dict = {}
            for feat in self.dataclass.feature_names:
                feat_vec = []
                for mc in user_metric_costs:
                    feat_vec.append(mc['costs'][feat])
                metric_dict[feat] = np.array(feat_vec)
                assert metric_dict[feat].dtype != np.object, "Seems like the length of cost vectors are different across samples."
            users_with_metric_info.append({'user_cost': user_cost, 'metric_costs': user_metric_costs})
            users.append({'user_cost': user_cost, 'metric_matrix': metric_dict})
        return users, users_with_metric_info

    # @time_profile(sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def sample_single_user_costs(self, user_id, user_alpha):
        user_dict = {}
        user_dict['alpha'] = user_alpha
        user_dict['editable_features'] = self.sample_features()
        user_dict['preference_scores'], user_dict['concentration'] = self.sample_preferences(user_dict['editable_features'])
        user_dict['valid_ranges'] = self.get_valid_ranges(user_dict['editable_features'], user_id)
        *_, user_dict[f'cost'] = self.get_cost(user_id, user_dict['preference_scores'], user_dict['valid_ranges'], alpha=user_alpha)
        for alph in np.arange(0, 1.1, 0.1):
            *_, user_dict[f'cost_{alph:.1f}'] = self.get_cost(user_id, user_dict['preference_scores'], user_dict['valid_ranges'], alpha=np.round(alph, 1))
        user_dict['query'] = self.pop_features[user_id]
        return user_dict

    def sample_single_user_metric_costs(self, user_id, user_alpha):
        metric_dict = {}
        metric_dict['alpha'] = user_alpha
        metric_dict['editable_features'] = self.sample_features()
        metric_dict['preference_scores'], metric_dict['concentration'] = self.sample_preferences(metric_dict['editable_features'])
        metric_dict['valid_ranges'] = self.get_valid_ranges(metric_dict['editable_features'], user_id)
        *_, metric_dict['costs'] = self.get_cost(user_id, metric_dict['preference_scores'], metric_dict['valid_ranges'], alpha=user_alpha)

        del metric_dict['valid_ranges']
        del metric_dict['editable_features']
        del metric_dict['preference_scores']

        return metric_dict

    def sample_features(self):
        non_fixed_features = self.od + self.uod + self.pod
        num_editable_features = np.random.randint(1, len(non_fixed_features)+1)
        # TODO: maybe decrease the prob of sampling a lot of features.
        editable_features = np.random.choice(non_fixed_features, num_editable_features, replace=False)
        return editable_features

    def sample_preferences(self, editable_features, eps=1e-7):
        # a = np.array([np.random.uniform(low=1, high=1) if feat in editable_features else np.random.uniform(low=0, high=0) for feat in self.dataclass.feature_names])
        # conc = np.array([np.random.uniform(low=min(a[idx], 1), high=1) if feat in editable_features else np.random.uniform(low=0, high=0) for idx, feat in enumerate(self.dataclass.feature_names)])
        conc = np.array([1 if feat in editable_features else eps for idx, feat in enumerate(self.dataclass.feature_names)])
        preference = np.random.dirichlet(conc)
        pd = {feat: preference[idx] if feat in editable_features else 0 for idx, feat in enumerate(self.dataclass.feature_names)}
        return pd, conc

    def kl_dirichlet(self, p, q, eps=1e-7):
        ''' p_i > 0, q_i > 0 for all i. '''
        return kl_divergence(Dirichlet(torch.tensor(p) + eps), Dirichlet(torch.tensor(q) + eps))

    def get_valid_ranges(self, editable_features, user_id, sample=False):
        sampled_range = copy.deepcopy(self.original_ranges)
        user_feat = self.pop_features[user_id]
        # for feat in editable_features:
        for feat in self.dataclass.feature_names:
            if self.dataclass.feature_change_restriction[feat] == -2 or feat not in editable_features:
                if feat in self.dataclass.continuous_feature_names:
                    sampled_range[feat] = [user_feat[feat], user_feat[feat]]
                elif feat in self.dataclass.categorical_feature_names:
                    sampled_range[feat] = [user_feat[feat]]
                continue

            elif self.dataclass.feature_change_restriction[feat] == 0:
                if feat in self.dataclass.categorical_feature_names:
                    restricted_range = self.dataclass.feature_vals[feat]
                elif feat in self.dataclass.continuous_feature_names:
                    restricted_range = [sampled_range[feat][0], sampled_range[feat][1]]

            elif self.dataclass.feature_change_restriction[feat] == 1:
                if self.dataclass.feature_types[feat] == 'ordered':
                    if feat in self.dataclass.categorical_feature_names:
                        restricted_range = self.dataclass.feature_vals[feat][self.dataclass.feature_vals[feat].index(user_feat[feat]): ]
                    elif feat in self.dataclass.continuous_feature_names:
                        restricted_range = [user_feat[feat], sampled_range[feat][1]]
                else:
                    raise ValueError('Feature is not ordered.')

            elif self.dataclass.feature_change_restriction[feat] == -1:
                if self.dataclass.feature_types[feat] == 'ordered':
                    if feat in self.dataclass.categorical_feature_names:
                        restricted_range = self.dataclass.feature_vals[feat][ :self.dataclass.feature_vals[feat].index(user_feat[feat])+1]
                    elif feat in self.dataclass.continuous_feature_names:
                        restricted_range = [sampled_range[feat][0], user_feat[feat]]
                else:
                    raise ValueError('Feature is not ordered.')

            if sample:
                raise NotImplementedError('Not Implemented, need the ranges from users and then we can take the'
                                          ' intersection and set the range')
            else:
                sampled_range[feat] = restricted_range
        return sampled_range

    # @time_profile(sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def get_cost_means(self, user_id, feat, cats, scores, alpha):
        if (alpha - 0) == 0:
            cost_mean, cost_var = self.get_linear_cost(user_id, feat, cats, scores)
        elif (alpha - 0) == 1:
            cost_mean, cost_var = self.get_percentile_cost(user_id, feat, cats, scores)
        else:
            linear_cost_mean, linear_cost_var = self.get_linear_cost(user_id, feat, cats, scores)
            perc_cost_mean, perc_cost_var = self.get_percentile_cost(user_id, feat, cats, scores)
            cost_mean = np.round(linear_cost_mean * alpha + perc_cost_mean * (1-alpha), 4)
            assert all((linear_cost_mean == self.max_cost) == (perc_cost_mean == self.max_cost)) \
                       and all((perc_cost_mean == self.max_cost) == (cost_mean == self.max_cost)), 'The cost should be prohibited at the same locations'
            cost_var = perc_cost_var
        return cost_mean, cost_var

    def get_percentile_cost(self, user_id, feat, cats, scores):
        aligned = False
        fv = self.pop_features[user_id][feat]
        assert all(not isinstance(elem, list) for elem in cats), 'Object should be a list of non-iterables.'
        if feat in self.dataclass.continuous_feature_names:
            org_range = self.cont_to_range(self.original_ranges[feat])
        elif feat in self.dataclass.categorical_feature_names:
            org_range = self.original_ranges[feat]
        cost_mean = [self.max_cost] * len(org_range)
        cost_var = [0] * len(org_range)
        flat_cats = cats

        if scores[feat] == 0 or len(cats) == 1 or self.dataclass.feature_types[feat] in ['fixed']:
            flat_mean = [0]
            pass
        elif self.dataclass.feature_types[feat] in ['ordered']:
            # (od, 1), (od,-1), (od,0), (f,-2)
            cat_order = list(self.dataclass.percentiles[feat].keys())
            feat_perc_idx = cat_order.index(fv)
            if self.dataclass.feature_change_restriction[feat] == 1:
                assert np.array_equal(cat_order[feat_perc_idx:], flat_cats), 'The categories should match.'
                flat_percentiles = list(self.dataclass.percentiles[feat].values())[feat_perc_idx:]
                flat_mean = np.abs(np.array(flat_percentiles) - self.dataclass.percentiles[feat][fv])
            elif self.dataclass.feature_change_restriction[feat] == -1:
                assert np.array_equal(cat_order[:feat_perc_idx+1], flat_cats), 'The categories should match.'
                flat_percentiles = list(self.dataclass.percentiles[feat].values())[:feat_perc_idx+1]
                flat_mean = np.abs(np.array(flat_percentiles) - self.dataclass.percentiles[feat][fv])
            elif self.dataclass.feature_change_restriction[feat] == 0:
                # ordered means diffrential cost, 0 means can go either ways.
                assert np.array_equal(cat_order, flat_cats), 'The categories should match.'
                flat_percentiles = list(self.dataclass.percentiles[feat].values())
                flat_mean = np.abs(np.array(flat_percentiles) - self.dataclass.percentiles[feat][fv])

            flat_mean = np.array(flat_mean) * (1 - scores[feat])
            if feat in self.dataclass.continuous_feature_names:
                aligned = True
                # feat_org_idx = org_range.index(fv)
                if self.dataclass.feature_change_restriction[feat] == 1:
                    cost_mean[feat_perc_idx:] = flat_mean
                    cost_var[feat_perc_idx:] = [self.variance] * len(flat_mean)
                elif self.dataclass.feature_change_restriction[feat] == -1:
                    cost_mean[:feat_perc_idx+1] = flat_mean
                    cost_var[:feat_perc_idx+1] = [self.variance] * len(flat_mean)
                elif self.dataclass.feature_change_restriction[feat] == 0:
                    cost_mean = flat_mean
                    cost_var = [self.variance] * len(flat_mean)

        elif self.dataclass.feature_types[feat] in ['unordered']:
            # unordered can only be 0
            flat_mean = np.random.uniform(0,1, len(cats)) #[(1 / (len(cats) - 1))] * len(cats) if len(cats) > 1 else [0]
            flat_mean[list(cats).index(fv)] = 0
            flat_mean = np.array(flat_mean) * (1 - scores[feat])

        if not aligned:
            for ii, c in enumerate(flat_cats):
                idx = org_range.index(c)
                cost_mean[idx] = flat_mean[ii]
                cost_var[idx] = self.variance
        cost_var[org_range.index(fv)] = 0
        cost_mean[org_range.index(fv)] = 0

        return np.round(np.array(cost_mean), 4), np.round(np.array(cost_var), 4)

    def get_linear_cost(self, user_id, feat, cats, scores):
        aligned = False
        fv = self.pop_features[user_id][feat]
        assert all(not isinstance(elem, list) for elem in cats), 'Object should be a list of non-iterables.'
        if feat in self.dataclass.continuous_feature_names:
            org_range = self.cont_to_range(self.original_ranges[feat])
        elif feat in self.dataclass.categorical_feature_names:
            org_range = self.original_ranges[feat]
        cost_mean = [self.max_cost] * len(org_range)
        cost_var = [0] * len(org_range)
        flat_cats = cats

        if scores[feat] == 0 or len(cats) == 1 or self.dataclass.feature_types[feat] in ['fixed']:
            flat_mean = [0]
        elif self.dataclass.feature_types[feat] in ['ordered']:
            # (od, 1), (od,-1), (od,0), (f,-2)
            if self.dataclass.feature_change_restriction[feat] == 1:
                flat_mean = np.linspace(0, 1, len(cats))
            elif self.dataclass.feature_change_restriction[feat] == -1:
                flat_mean = np.linspace(0, 1, len(cats))[::-1]
            elif self.dataclass.feature_change_restriction[feat] == 0:
                # ordered means diffrential cost, 0 means can go either ways.
                post = np.linspace(0, 1, len(cats[cats.index(fv):])).tolist()
                pre = np.linspace(0, 1, len(cats[:cats.index(fv) + 1]))[::-1].tolist()
                flat_mean = pre[:-1] + post
            flat_mean = np.array(flat_mean) * (1 - scores[feat])
            assert len(flat_mean) == len(cats), 'Lengths have to be the same.'

            if feat in self.dataclass.continuous_feature_names:
                aligned = True
                feat_org_idx = org_range.index(fv)
                if self.dataclass.feature_change_restriction[feat] == 1:
                    assert np.array_equal(org_range[feat_org_idx:], cats), 'Must be equal.'
                    cost_mean[feat_org_idx:] = flat_mean
                    cost_var[feat_org_idx:] = [self.variance] * len(flat_mean)
                elif self.dataclass.feature_change_restriction[feat] == -1:
                    assert np.array_equal(org_range[:feat_org_idx+1], cats), 'Must be equal.'
                    cost_mean[:feat_org_idx+1] = flat_mean
                    cost_var[:feat_org_idx+1] = [self.variance] * len(flat_mean)
                elif self.dataclass.feature_change_restriction[feat] == 0:
                    assert np.array_equal(org_range, cats), 'Must be equal.'
                    cost_mean = flat_mean
                    cost_var = [self.variance] * len(flat_mean)
                # cost_var[feat_org_idx] = 0

        elif self.dataclass.feature_types[feat] in ['unordered']:
            # unordered can only be 0
            flat_mean = np.random.uniform(0, 1, len(cats)) #[(1 / (len(cats) - 1))] * len(cats) if len(cats) > 1 else [0]
            flat_mean[list(cats).index(fv)] = 0
            flat_mean = np.array(flat_mean) * (1 - scores[feat])

        if not aligned:
            for ii, c in enumerate(flat_cats):
                idx = org_range.index(c)
                cost_mean[idx] = flat_mean[ii]
                cost_var[idx] = self.variance
        cost_var[org_range.index(fv)] = 0
        cost_mean[org_range.index(fv)] = 0

        return np.round(np.array(cost_mean), 4), np.round(np.array(cost_var), 4)

    def cont_to_range(self, range):
        return np.arange(range[0], range[1]+1).tolist()

    def sample_cost(self, means, vars, eps=1e-7):
        samples = np.ones_like(means) * self.max_cost
        zero_idx = np.where(means == 0)[0]
        pos_idx = np.where((means <= 1) & (means > 0))[0]
        means_pos = means[pos_idx] + eps
        means_pos[means_pos >= 1] = 1 - eps * 1.1
        vars_pos = vars[pos_idx] + eps
        alphas = ((1-means_pos)/vars_pos - 1/means_pos) * means_pos**2
        alphas[alphas < 0] = eps
        betas = alphas * (1/means_pos - 1)
        samples[pos_idx] = np.random.beta(alphas, betas)
        samples[zero_idx] = 0
        return samples

    def get_cost(self, user_id, preference_scores, valid_ranges, alpha):
        mean_cost_dict, var_cost_dict, cost_dict = {}, {}, {}
        for feat in self.dataclass.feature_names:
            valid_feat_range = valid_ranges[feat]
            if feat in self.dataclass.continuous_feature_names:
                valid_feat_range = self.cont_to_range(valid_feat_range)
            cost_means, cost_vars = self.get_cost_means(user_id, feat, valid_feat_range, preference_scores, alpha=alpha)
            cost = self.sample_cost(cost_means, cost_vars)
            mean_cost_dict[feat] = cost_means
            var_cost_dict[feat] = cost_vars
            cost_dict[feat] = cost
        # print(f'User {user_id}')
        return mean_cost_dict, var_cost_dict, cost_dict

class DataIO(object):

    def __init__(self, args):
        self.hparams = args
        self.data_name = args.data_name
        self.project_dir = args.project_dir
        self.dataset_folder = os.path.join(self.project_dir, args.data_folder, 'final')

        if args.dump_negative_data:
            from utils.helpers import setup_model, setup_pandas
            from train_model import prepare_data_for_training

            self.dataset = pickle.load(open(f'{self.dataset_folder}/{self.data_name}/processed_data.pkl', 'rb'))
            self.create_pandas_splits()
            pandas_data_obj = setup_pandas(self)
            pandas_data_obj.args = args

            model, predict_fn = setup_model(args)
            x = prepare_data_for_training(pandas_data_obj, split='test')[0]
            preds = np.argmax(
                model.predict_probs(prepare_data_for_training(pandas_data_obj, split='test')[0], return_last=False),
                axis=1)
            test_idx_negative = self.dataset.test_idx[np.where(preds == 0)[0]]
            test_idx_positive = self.dataset.test_idx[np.where(preds != 0)[0]]
            assert all([i in self.dataset['test_idx'] for i in test_idx_negative])
            # self.dataset['train_idx'] = np.hstack((self.dataset['train_idx'], test_idx_positive))
            self.dataset['test_idx'] = test_idx_negative
            fname = f'{self.dataset_folder}/{self.data_name}/processed_data_tnfn.pkl'
            print(f'Dumping TN + FN at {fname}')
            pickle.dump(self.dataset, open(fname, 'wb'))

        if args.save_data:
            from utils.helpers import setup_pandas

            self.dataset = pickle.load(open(f'{self.dataset_folder}/{self.data_name}/processed_data_tnfn.pkl', 'rb'))
            self.create_pandas_splits()
            pandas_data_obj = setup_pandas(self)
            pandas_data_obj.args = args

            user_obj = UserSampler(pandas_data_obj, args)

            self.dataset.users_info = user_obj.users_info
            if not args.debug:
                self.dump_data(args.alpha, args.num_mcmc)

            self.dataset.users_info = user_obj.users_info_with_metric_info
            if not args.debug:
                self.dump_data(args.alpha, args.num_mcmc, 'dump_all')
        else:
            if 'batch_size' in args:
                self.load_train_data()
            else:
                self.load_cost_data(args.alpha, args.num_mcmc, args.load_type)
            self.create_pandas_splits()

    def create_pandas_splits(self):
        self.data = self.convert_to_pandas('all')
        self.data_train = self.convert_to_pandas('train')
        self.data_val = self.convert_to_pandas('val')
        self.data_test = self.convert_to_pandas('test')

    def dump_data(self, alpha, num_mcmc, type=''):
        if alpha != None:
            save_dir = os.path.join(self.project_dir, 'data', 'cost', f'mcmc{num_mcmc}', f'{alpha:.1f}')
        else:
            save_dir = os.path.join(self.project_dir, 'data', 'cost', f'mcmc{num_mcmc}', 'random_alphas')
        os.makedirs(save_dir, exist_ok=True)
        if type != '':
            fname = f'{self.data_name}_all.pkl'
        else:
            fname = f'{self.data_name}.pkl'

        for name, data in zip(['train','dev','test'], [self.dataset.train_idx, self.dataset.val_idx, self.dataset.test_idx]):
            print('Dumping %s data: %d samples.' % (name, len(data)))
        print(f"Dumping data: {os.path.join(save_dir, f'{fname}')}")
        pickle.dump(self.dataset, open(os.path.join(save_dir, f'{fname}'), 'wb'))

    def load_train_data(self):
        print(f"Loading data: {f'{self.dataset_folder}/{self.data_name}/processed_data.pkl'}")
        self.dataset = pickle.load(open(f'{self.dataset_folder}/{self.data_name}/processed_data.pkl', 'rb'))

    def load_cost_data(self, alpha, num_mcmc, load_type=''):
        if alpha != None:
            save_dir = os.path.join(self.project_dir, 'data', 'cost', f'mcmc{num_mcmc}', f'{alpha:.1f}')
        else:
            save_dir = os.path.join(self.project_dir, 'data', 'cost', f'mcmc{num_mcmc}', 'random_alphas')
        if load_type != '':
            fname = f'{self.data_name}_all.pkl'
        else:
            fname = f'{self.data_name}.pkl'

        print(f"Loading data: {os.path.join(save_dir, f'{fname}')}")
        self.dataset = pickle.load(open(os.path.join(save_dir, f'{fname}'), 'rb'))

        for name, data in zip(['train','dev','test'], [self.dataset.train_idx,self.dataset.val_idx,self.dataset.test_idx]):
            print('Loading %s: %d samples.' % (name, len(data)))

    def convert_to_pandas(self, split='all'):
        if split == 'all':
            arr = self.dataset.data
            target = self.dataset.labels
        elif split == 'train':
            arr = self.dataset.data[self.dataset.train_idx]
            target = self.dataset.labels[self.dataset.train_idx]
        elif split == 'val':
            arr = self.dataset.data[self.dataset.val_idx]
            target = self.dataset.labels[self.dataset.val_idx]
        elif split == 'test':
            arr = self.dataset.data[self.dataset.test_idx]
            target = self.dataset.labels[self.dataset.test_idx]
        df = pd.DataFrame.from_records(arr, columns=self.dataset.feature_names)
        df[self.dataset.class_target] = target
        df[self.dataset.ordinal_fnames] = df[self.dataset.ordinal_fnames].astype('int')
        df = df.reset_index(drop=True, inplace=False)
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data IO params
    parser.add_argument('--data_name', default=['adult_binary'], nargs="+")
    parser.add_argument('--project_dir', default='../', type=str)
    parser.add_argument('--data_folder', default='data', type=str, help='Name of the folder containing data inside project Dir.')
    parser.add_argument('--save_data', action='store_true', help='Save the data splits.')
    parser.add_argument('--dump_negative_data', action='store_true', help='Test the test data and dump only negative class in test.')
    parser.add_argument('--dist_samples', action='store_true', help='Dump distribution shift costs.')
    parser.add_argument('--alpha_data', action='store_true', help='Dump data for all alphas.')
    parser.add_argument('--load_type', default='', type=str)

    # Cost sampling
    parser.add_argument('--alpha', default=None, type=float, help='multiplicative factor of linear cost, 0=linear, 1=percentile, frac-convex combination.')
    parser.add_argument('--num_mcmc', default=500, type=int)
    # Experiment params
    parser.add_argument('--num_users', default=3, type=int, help='number of counterfactual required.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--backend', default='PYT', type=str)
    # General params
    parser.add_argument('--seed', default=-1, type=int, help='seed for np and random.')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    print(f'Creating user for {args.data_name}')

    if args.seed == -1:
        args.seed = np.random.randint(1, 1000)
        print(f'RANDOM SEED: {args.seed}')

    names = args.data_name
    for name in names:
        args.data_name = name
        DataIO(args)
        if args.alpha_data:
            args.dump_negative_data = False
            for a in np.arange(0, 1.05, 0.2):
                args.alpha = np.round(a, 1)
                DataIO(args)

    pass


