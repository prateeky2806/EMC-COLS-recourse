
"""
Module to generate diverse counterfactual explanations based on random sampling.
A simple implementation.
"""
from explainers.explainer_base import ExplainerBase
import numpy as np
import pandas as pd
import random
import timeit
import torch
import copy
from itertools import product

from loaders import counterfactuals_example as cf_exp
from utils.metrics import MetricsEvaluator, UserEvaluator

class BaseSearch(ExplainerBase):

    def __init__(self, data_interface, model_interface, evaluator):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        sample_sets: sample sets of size k, and select the one which performs the best on metrics.
        """
        super().__init__(data_interface)  # initiating data related parameters
        # Load and set the model_loaders in evaluation mode
        self.model = model_interface
        self.model.load_model()
        ev = self.model.set_eval_mode()
        self.evaluator = evaluator

        temp_input = torch.rand([len(self.data_interface.encoded_feature_names)]).float()
        self.num_output_nodes = len(self.predict_probs(temp_input, return_last=False, return_numpy=True))

        self.precisions = self.data_interface.get_decimal_precisions(output_type="dict")
        if self.data_interface.outcome_name in self.precisions:
            self.outcome_precision = [self.precisions[self.data_interface.outcome_name]]
        else:
            self.outcome_precision = 0

    def setup_query_params(self):
        self.query_numpy = self.data_interface.prepare_query_instance(query_instance=self.query, encode=True).iloc[0].values
        self.query_df = pd.DataFrame(dict(zip(self.query.keys(), [[q] for q in self.query.values()])), columns=self.data_interface.feature_names)
        self.query_pred = self.predict_probs(self.query_numpy, return_last=True, return_numpy=True)
        self.query_class = np.round(self.query_pred)
        self.target_cf_class = self.infer_target_cfs_class(self.desired_class, self.query_pred, self.num_output_nodes)

    def set_ranges_and_features_to_vary(self):
        if self.permitted_range is None:
            self.feature_range = self.data_interface.permitted_range
        else:
            self.feature_range, feature_ranges_orig = self.data_interface.get_features_range(self.permitted_range)

        self.features_to_vary = self.features_to_vary
        if self.features_to_vary == "all":
            self.features_to_vary = self.data_interface.feature_names
            self.fixed_features_values = {}
        else:
            self.fixed_features_values = {}
            for feature in self.data_interface.feature_names:
                if feature not in self.features_to_vary:
                    self.fixed_features_values[feature] = self.query[feature].iat[0]

    def post_process(self, cfs_df):
        self.valid_cfs_found = True if self.total_cfs_found >= self.total_CFs else False
        cfs_df[self.data_interface.outcome_name] = self.get_model_output_from_scores(self.cfs_pred_scores)
        final_cfs_df = cfs_df[self.data_interface.feature_names + [self.data_interface.outcome_name]]
        final_cfs_df[self.data_interface.outcome_name] = final_cfs_df[self.data_interface.outcome_name].round(self.outcome_precision)
        self.cfs_preds = final_cfs_df[[self.data_interface.outcome_name]].values
        self.final_cfs = final_cfs_df[self.data_interface.feature_names].values
        self.final_cfs_encoded = self.data_interface.transform_data(final_cfs_df.copy()[self.data_interface.feature_names],
                                                                    encode=True, normalise=True, return_numpy=True)
        self.final_cfs_sparse, self.cfs_preds_sparse = None, None

    def get_score_and_metrics(self, cfs_df, do_avg=True, return_preds=False):
        tmp_pred_scores, tmp_pred_classes, tmp_pred_validity = self.get_predictions_class_and_validity_from_df(cfs_df)
        if isinstance(self.evaluator, MetricsEvaluator):
            metrics, names = self.evaluator.get_single_sample_metrics(cfs_df.values.astype('<U32'), self.query_df.values, self.query_class,
                                                                      tmp_pred_classes.flatten(), type=self.eval, return_dict=True)
            if return_preds:
                return self.evaluator.get_score_from_metric(metrics, type=self.eval), metrics, tmp_pred_scores, tmp_pred_classes, tmp_pred_validity
            else:
                return self.evaluator.get_score_from_metric(metrics, type=self.eval), metrics
        elif isinstance(self.evaluator, UserEvaluator):
            costs, validity_mask = self.evaluator.get_single_sample_cost(cfs_df.values.astype('<U32'), self.user_id, self.query_class, tmp_pred_classes.flatten(), mcmc=True)
            assert np.array_equal(validity_mask, tmp_pred_validity), "The validity masks should be same."
            if return_preds:
                return self.evaluator.get_score_from_cost(costs, do_avg), costs, tmp_pred_scores, tmp_pred_classes, tmp_pred_validity
            else:
                return self.evaluator.get_score_from_cost(costs, do_avg), costs

    def get_samples_predictions_class_and_validity(self, sample_size):
        samples = self.get_samples(sampling_random_seed=None, sampling_size=sample_size)
        probs, classes, validity = self.get_predictions_class_and_validity_from_df(samples)
        return samples, probs, classes, validity

    def get_predictions_class_and_validity_from_df(self, samples):
        predict_data = self.data_interface.transform_data(samples, encode=True, normalise=True, return_numpy=True)
        probs = self.predict_probs(predict_data, return_last=True, return_numpy=True)
        classes = np.round(probs)
        validity = self.decide_cf_validity(probs)
        return probs, classes, validity.astype(bool)

    def initialise_cfs(self, init_type, sample_size, seed=None):
        if init_type == 'random':
            samples = self.get_samples(sampling_size=sample_size, sampling_random_seed=seed)
        elif 'ham' in init_type:
            order = int(init_type.split('_')[-1])
            samples = copy.deepcopy(self.query_df)
            samples = samples.append([samples] * (sample_size-1), ignore_index=True)
            for i in range(sample_size):
                features = np.random.choice(self.data_interface.feature_names, order, replace=False)
                for feat in features:
                    if feat in self.data_interface.continuous_feature_names:
                        low = self.feature_range[feat][0]
                        high = self.feature_range[feat][1]
                        samples.loc[i, feat] = self.get_continuous_samples(low, high, self.precisions[feat], size=1, seed=None)[0]
                    elif feat in self.data_interface.categorical_feature_names:
                        cat_choice = copy.deepcopy(self.data_interface.categorical_map_str2liststr[feat])
                        cat_choice.remove(samples.loc[i, feat])
                        samples.loc[i, feat] = np.random.choice(cat_choice)
        elif 'org' in init_type:
            samples = copy.deepcopy(self.query_df)
            samples = samples.append([samples] * (sample_size - 1), ignore_index=True)
        else:
            raise NotImplementedError('init_type can only be random, ham_x or org')
        samples.reset_index(inplace=True, drop=True)
        return samples

    def get_perturbed_cfs(self, input_cfs, perturb_type, hamming_dist=2):
        cfs = input_cfs.copy().reset_index(drop=True)
        if perturb_type == 'one':
            num_edits = 1
        elif 'frac' in perturb_type:
            num_edits = (len(cfs) * int(perturb_type.split('_')[1]))//10
        elif perturb_type == 'all':
            num_edits = len(cfs)
        else:
            raise NotImplementedError(f'perturb_type has to be in ["one", "frac_x", "all"]')
        row_idx = np.random.choice(np.arange(len(cfs)), size=num_edits, replace=False)
        for ii, row in enumerate(row_idx):
            row_df = cfs.loc[row:row].copy().reset_index(drop=True)
            # row_df.reset_index(inplace=True, drop=True)
            probs_all = [1,1,0.4,0.2, 0.1] + [0.05]*30
            probs = np.array(probs_all[:hamming_dist])
            probs = probs / probs.sum(keepdims=True)
            num_feats_to_edit = np.random.choice(np.arange(1, hamming_dist+1), p=probs)
            feat_to_edit = np.random.choice(self.data_interface.feature_names, size=num_feats_to_edit, replace=False)
            for feat in feat_to_edit:
                map_cat_choice = copy.deepcopy(self.data_interface.categorical_map_str2liststr)
                if feat in self.data_interface.categorical_feature_names:
                    map_cat_choice[feat].remove(row_df.loc[0, feat])
                    row_df.loc[0, feat] = np.random.choice(map_cat_choice[feat])
                elif feat in self.data_interface.continuous_feature_names:
                    low = self.feature_range[feat][0]
                    high = self.feature_range[feat][1]
                    row_df.loc[0, feat] = self.get_continuous_samples(low, high, self.precisions[feat], size=1, seed=None)[0]
            cfs.loc[row] = row_df.values[0]
        return cfs

    def update_cfs(self, cfs, perturb_type, hamming_dist, do_avg):
        cfs_temp = self.get_perturbed_cfs(cfs.copy(), perturb_type, hamming_dist)
        if self.eval == 'cost_simple':
            score_temp, metrics_temp, pred_scr, pred_class, validity = self.get_score_and_metrics(cfs_temp, do_avg=True, return_preds=True)
        else:
            score_temp, metrics_temp, pred_scr, pred_class, validity = self.get_score_and_metrics(cfs_temp, do_avg, return_preds=True)
        return cfs_temp, score_temp, metrics_temp, pred_scr, pred_class, validity

    def get_unique(self, samples, metrics, pred_scr, pred_class):
        assert len(pred_scr) == len(samples) == len(metrics) == len(
            pred_class), "Length mismatch between validity, metrics and cfs"
        unique_rows = (~samples.duplicated(keep=False)).values
        cfs_unique = samples.loc[unique_rows].reset_index(drop=True)
        # score_unique = score[unique_rows]
        metrics_unique = metrics[unique_rows]
        pred_scr_unique = pred_scr[unique_rows]
        pred_class_unique = pred_class[unique_rows]
        return cfs_unique, metrics_unique, pred_scr_unique, pred_class_unique

    def cfs_to_add(self, best_metrics, curr_metrics):
        try:
            benefit = best_metrics.min(0) - curr_metrics
        except:
            import pdb; pdb.set_trace()
        benefit = np.nansum(np.where(benefit > 0, benefit, np.nan), axis=1)
        top_indices = benefit[benefit > 0].argsort()[::-1]
        return top_indices

    def cfs_to_replace(self, best_metrics, curr_metrics):
        best_idx_per_cost = best_metrics.argsort(0)[0]
        second_best_idx_per_cost = best_metrics.argsort(0)[1]
        ben_mat = np.zeros((len(best_metrics), len(curr_metrics)))
        for bb, best_met in enumerate(best_metrics):
            for cc, curr_met in enumerate(curr_metrics):
                ben = 0
                for j in np.where(best_idx_per_cost == bb)[0]:
                    if best_met[j] > curr_met[j]:
                        ben += (best_met[j] - curr_met[j])
                    else:
                        ben += (best_met[j] - min(curr_met[j], best_metrics[second_best_idx_per_cost[j]][j]))
                ben_mat[bb, cc] = ben
        if (ben_mat > 0).sum() == 0:
            return []
        else:
            # TODO: This pair selection procedure can be improved.
            past = []
            new = []
            for cc in range(len(curr_metrics)):
                for idx in ben_mat[:, cc].argsort()[::-1]:
                    if idx not in past and ben_mat[idx, cc] > 0:
                        past.append(idx)
                        new.append(cc)
            return list(zip(past, new))

    # @time_profile(sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def cost_local_search(self, perturb_type='all', init_type='ham_2', hamming_dist=2):
        self.model.local_forward = 0
        validity = [0]
        itr = 0
        seeds = np.random.randint(0,1000, 10000)
        while sum(validity) <= 0:
            samples = self.initialise_cfs(init_type, self.total_CFs, seed=seeds[itr])
            # if itr>1:
            #     import pdb; pdb.set_trace()
            (min_per_mcmc, cf_idx_mcmc), metrics, pred_scr, pred_class, validity = self.get_score_and_metrics(samples, do_avg=False, return_preds=True)
            score = min_per_mcmc.mean()
            best_samples_tmp = samples.copy()
            # best_samples, best_score = samples.copy().reset_index(drop=True), score
            # best_metrics, best_pred_scr, best_pred_class = metrics, pred_scr, pred_class
            best_samples, best_score = samples.loc[validity].copy().reset_index(drop=True), score
            best_metrics, best_pred_scr, best_pred_class = metrics[validity], pred_scr[validity], pred_class[validity]
            itr += 1
            if itr >= 10:
                init_type = 'random'
            if itr > 15:
                print('Could not find a valid initialisation')
                break

        while best_score > 0 and self.model.local_forward < self.local_budget:
            if self.iter_type == 'best':
                samples, (min_per_mcmc, cf_idx_mcmc), metrics, pred_scr, pred_class, validity = self.update_cfs(best_samples_tmp, perturb_type,
                                                                                          hamming_dist, do_avg=False)
            elif self.iter_type == 'linear':
                samples, (min_per_mcmc, cf_idx_mcmc), metrics, pred_scr, pred_class, validity = self.update_cfs(samples, perturb_type,
                                                                                          hamming_dist, do_avg=False)
            else:
                raise NotImplementedError(f'iter_type can only be "best" or "linear", given {self.iter_type}')

            if sum(validity) > 0:
                if len(best_samples) < self.total_CFs:
                    idx_priority = self.cfs_to_add(best_metrics, metrics[validity])
                    if len(idx_priority) >= self.total_CFs - len(best_samples):
                        idx_to_add = idx_priority[:self.total_CFs - len(best_samples)]
                    else:
                        idx_to_add = idx_priority
                    best_samples = best_samples.append(samples[validity].iloc[idx_to_add], ignore_index=True)
                    best_metrics = np.vstack([best_metrics, metrics[validity][idx_to_add]])
                else:
                    assert len(best_samples) == self.total_CFs
                    if len(best_metrics) == 1 and len(metrics[validity]) == 1:
                        if best_metrics.mean() > metrics[validity].mean():
                            best_samples = samples[validity]
                            best_metrics = metrics[validity]
                    else:
                        replace_idx = self.cfs_to_replace(best_metrics, metrics[validity])
                        for pair in replace_idx:
                            best_samples.iloc[pair[0]] = samples[validity].iloc[pair[1]]
                            best_metrics[pair[0]] = metrics[validity][pair[1]]
                best_samples_tmp = best_samples_tmp.append(best_samples, ignore_index=True).tail(self.total_CFs)
        # if not enough valid samples then append with random samples.
        if len(best_samples) < self.total_CFs:
            random_cfs = self.get_samples(sampling_size=self.total_CFs-len(best_samples), sampling_random_seed=None)
            best_samples = best_samples.append(random_cfs, ignore_index=True)
        best_score, best_metrics, best_pred_scr, best_pred_class, best_validity = self.get_score_and_metrics(best_samples, do_avg=True, return_preds=True)

        return best_samples, best_score, best_metrics, best_pred_scr, best_pred_class, best_validity

    # @time_profile(sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def local_search(self, perturb_type='all', init_type='ham_2', hamming_dist=2):
        self.model.local_forward = 0
        validity = [0]
        itr = 0
        seeds = np.random.randint(0,1000, 10000)
        while sum(validity) <= 0:
            samples = self.initialise_cfs(init_type, self.total_CFs, seed=seeds[itr])
            if self.eval == 'cost_simple':
                score, metrics, pred_scr, pred_class, validity = self.get_score_and_metrics(samples, do_avg=True, return_preds=True)
            else:
                score, metrics, pred_scr, pred_class, validity = self.get_score_and_metrics(samples, do_avg=None, return_preds=True)
            best_samples, best_score = samples.copy().reset_index(drop=True), score

            itr += 1
            if itr >= 10:
                init_type = 'random'
            if itr > 15:
                print('Could not find a valid initialisation')
                break

        while self.model.local_forward < self.local_budget:
            samples, score, metrics, pred_scr, pred_class, validity = self.update_cfs(best_samples.copy(), perturb_type, hamming_dist, do_avg=None)
            if score < best_score:
                best_samples, best_score = samples, score

        assert len(best_samples) == self.total_CFs, f'The output needs to contain {self.total_CFs} samples'
        if self.eval == 'cost_simple':
            best_score, best_metrics, best_pred_scr, best_pred_class, best_validity = self.get_score_and_metrics(best_samples, do_avg=True, return_preds=True)
        else:
            best_score, best_metrics, best_pred_scr, best_pred_class, best_validity = self.get_score_and_metrics(best_samples, do_avg=None, return_preds=True)
        return best_samples, best_score, best_metrics, best_pred_scr, best_pred_class, best_validity

    # @time_profile(sort_by='cumulative', lines_to_print=20, strip_dirs=True)
    def parallel_local_search(self, num_parallel_runs, perturb_type='one', init_type='random', hamming_dist=2):
        cfs_cache, scr_cache, metrics_cache, pred_scr_cache, pred_class_cache, validity_cache = [], [], [], [], [], []
        # from joblib import Parallel, delayed
        # parallel_results = Parallel(n_jobs=num_parallel_runs)(delayed(self.local_search)(perturb_type, init_type, hamming_dist) for i in range(num_parallel_runs))
        # for res in parallel_results:
        #     cfs_cache.append(res[0])
        #     metrics_cache.append(res[1])
        #     pred_scr_cache.append(res[2])
        #     pred_class_cache.append(res[3])
        for run in range(num_parallel_runs):
            # print(f"Run {run}/{num_parallel_runs-1}")
            if self.eval == 'cost':
                cfs, score, metrics, pred_scr, pred_class, validity = self.cost_local_search(
                    perturb_type, init_type, hamming_dist)
            else:
                cfs, score, metrics, pred_scr, pred_class, validity = self.local_search(
                    perturb_type, init_type, hamming_dist)
            cfs_cache.append(cfs); metrics_cache.append(metrics); pred_scr_cache.append(pred_scr)
            pred_class_cache.append(pred_class); scr_cache.append(score); validity_cache.append(validity)

        best_scr_idx = np.array(scr_cache).argmin()
        return cfs_cache[best_scr_idx], scr_cache[best_scr_idx], metrics_cache[best_scr_idx], pred_scr_cache[best_scr_idx], pred_class_cache[best_scr_idx], validity_cache[best_scr_idx]

    def get_samples(self, sampling_random_seed=None, sampling_size=1):

        categorical_features_frequencies = {}
        for feature in self.data_interface.categorical_feature_names:
            categorical_features_frequencies[feature] = len(self.data_interface.data_df[feature].value_counts())

        if sampling_random_seed is not None:
            random.seed(sampling_random_seed)

        samples = []
        for feature in self.data_interface.feature_names:
            if feature in self.fixed_features_values:
                sample = [self.fixed_features_values[feature]]*sampling_size
            elif feature in self.data_interface.continuous_feature_names:
                low = self.feature_range[feature][0]
                high = self.feature_range[feature][1]
                sample = self.get_continuous_samples(low, high, self.precisions[feature], size=sampling_size, seed=sampling_random_seed)
            else:
                if sampling_random_seed is not None:
                    random.seed(sampling_random_seed)
                sample = random.choices(self.feature_range[feature], k=sampling_size)

            samples.append(sample)
        samples = pd.DataFrame(dict(zip(self.data_interface.feature_names, samples))) #to_dict(orient='records')#.values
        return samples

    def get_continuous_samples(self, low, high, precision, size=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if precision == 0:
            result = np.random.randint(low, high+1, size).tolist()
            result = [float(r) for r in result]
        else:
            result = np.random.uniform(low, high+(10**-precision), size)
            result = [round(r, precision) for r in result]
        return result

    def print_end_stats(self):
        m, s = divmod(self.elapsed, 60)
        if self.valid_cfs_found:
            if self.verbose:
                print('Diverse Counterfactuals found! total time taken: %02d' % m, 'min %02d' % s, 'sec')
        else:
            if self.total_cfs_found == 0 :
                print('No Counterfactuals found for the given configuration', '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) Diverse Counterfactuals found for the given configuration' % (self.total_cfs_found, self.total_CFs), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

    def predict_probs(self, input_instance, return_last=True, return_numpy=True):
        """Works 1d or 2d with numpy array or torch tensor
        Returns: np.array or torch.tensor of shape N(num_samples) x c (number of classes) or N x 1.
        """
        if torch.is_tensor(input_instance):
            input_instance = input_instance.float()
        else:
            input_instance = torch.tensor(input_instance.astype(np.float)).float()

        if len(input_instance.shape) == 1:
            ret = self.model.get_output(input_instance.unsqueeze(0))[0]
            if return_last:
                ret = ret[..., -1]
        elif len(input_instance.shape) == 2:
            ret = self.model.get_output(input_instance)
            if return_last:
                ret = ret[..., -1].unsqueeze(1)
        return ret.data.numpy() if return_numpy else ret


class RandomSearch(BaseSearch):
    def __init__(self, data_interface, model_interface, evaluator):
        super().__init__(data_interface, model_interface, evaluator)  # initiating data related parameters

    def random_search(self):
        self.model.local_forward = 0
        validity = [0]
        itr = 0
        seeds = np.random.randint(0, 1000, 10000)
        while sum(validity) <= 0:
            samples = self.initialise_cfs('random', self.total_CFs, seed=seeds[itr])
            score, metrics, pred_scr, pred_class, validity = self.get_score_and_metrics(samples, do_avg=True, return_preds=True)
            best_samples, best_score = samples.copy().reset_index(drop=True), score
            itr += 1
            if itr > 15:
                print('Could not find a valid initialisation')
                break

        while self.model.local_forward < self.budget:
            samples = self.get_samples(sampling_random_seed=None, sampling_size=self.total_CFs)
            score, metrics, pred_scr, pred_class, validity = self.get_score_and_metrics(samples, do_avg=True, return_preds=True)

            if score < best_score:
                best_samples, best_score = samples, score

        assert len(best_samples) == self.total_CFs, f'The output needs to contain {self.total_CFs} samples'
        best_score, best_metrics, best_pred_scr, best_pred_class, best_validity = self.get_score_and_metrics(best_samples, do_avg=True, return_preds=True)
        return best_samples, best_score, best_metrics, best_pred_scr, best_pred_class, best_validity

    def generate_counterfactuals(self, query_instance, verbose=False, **kwargs):
        """Generate counterfactuals by randomly sampling features."""
        self.stopping_threshold = 0.5
        self.query = query_instance
        self.verbose = verbose
        self.random_seed = kwargs['seed']
        self.total_CFs = kwargs['num_cfs']
        # np.random.seed(self.random_seed)
        self.permitted_range = kwargs['permitted_range'] if 'permitted_range' in kwargs.keys() else None
        self.features_to_vary = kwargs['features_to_vary'] if 'features_to_vary' in kwargs.keys() else 'all'
        self.desired_class = kwargs['desired_class'] if 'desired_class' in kwargs.keys() else 'opposite'
        self.user_id = kwargs['user_id']
        self.thresh = kwargs['thresh']
        self.budget = kwargs['budget']
        self.iter_type = kwargs['iter_type']
        self.eval = kwargs['eval']

        self.set_ranges_and_features_to_vary()
        self.precisions = self.data_interface.get_decimal_precisions(output_type="dict")
        self.setup_query_params()

        np.random.seed(self.random_seed)

        start_time = timeit.default_timer()
        cfs_df, score, metrics, self.cfs_pred_scores, self.cfs_pred_classes, validity = self.random_search()
        cfs_df.reset_index(inplace=True, drop=True)

        self.total_cfs_found = sum(validity)
        self.post_process(cfs_df)

        self.elapsed = timeit.default_timer() - start_time
        self.print_end_stats()

        return cf_exp.CounterfactualExamples(data_interface=self.data_interface,
                                             test_instance=self.query_numpy,
                                             test_pred=self.query_pred,
                                             final_cfs=self.final_cfs_encoded[:,np.newaxis,:],
                                             final_cfs_preds=self.cfs_preds,
                                             final_cfs_sparse=self.final_cfs_sparse[:,np.newaxis,:] if self.final_cfs_sparse is not None else self.final_cfs_sparse,
                                             cfs_preds_sparse=self.cfs_preds_sparse,
                                             posthoc_sparsity_param=None,
                                             desired_class=self.desired_class,
                                             num_forward_pass=self.model.num_forward_pass)


class LocalSearch(BaseSearch):

    def __init__(self, data_interface, model_interface, evaluator):
        super().__init__(data_interface, model_interface, evaluator)  # initiating data related parameters

    def generate_counterfactuals(self, query_instance, verbose=False, **kwargs):
        """Generate counterfactuals by randomly sampling features."""
        self.stopping_threshold = 0.5
        self.query = query_instance
        self.verbose = verbose
        self.random_seed = kwargs['seed']
        self.total_CFs = kwargs['num_cfs']
        np.random.seed(self.random_seed)
        self.permitted_range = kwargs['permitted_range'] if 'permitted_range' in kwargs.keys() else None
        self.features_to_vary = kwargs['features_to_vary'] if 'features_to_vary' in kwargs.keys() else 'all'
        self.desired_class = kwargs['desired_class'] if 'desired_class' in kwargs.keys() else 'opposite'
        perturb_type = kwargs['perturb_type'] if 'perturb_type' in kwargs.keys() else 'one'
        init_type = kwargs['init_type'] if 'init_type' in kwargs.keys() else 'random'
        hamming_dist = kwargs['hamming_dist'] if 'hamming_dist' in kwargs.keys() else 2
        self.user_id = kwargs['user_id']
        self.thresh = kwargs['thresh']
        self.local_budget = kwargs['budget']
        self.iter_type = kwargs['iter_type']
        self.eval = kwargs['eval']

        self.set_ranges_and_features_to_vary()
        self.precisions = self.data_interface.get_decimal_precisions(output_type="dict")

        # Do predictions once on the query_instance and reuse across to reduce the number inferences.
        self.setup_query_params()

        start_time = timeit.default_timer()
        if self.eval == 'cost':
            cfs_df, score, metrics, self.cfs_pred_scores, self.cfs_pred_classes, validity = self.cost_local_search(perturb_type, init_type, hamming_dist)
        else:
            cfs_df, score, metrics, self.cfs_pred_scores, self.cfs_pred_classes, validity = self.local_search(perturb_type, init_type, hamming_dist)
        # cfs_df = cfs_df.loc[validity].reset_index(drop=True)
        self.total_cfs_found = sum(validity)
        self.post_process(cfs_df)

        self.elapsed = timeit.default_timer() - start_time
        self.print_end_stats()
        return cf_exp.CounterfactualExamples(data_interface=self.data_interface,
                                             test_instance=self.query_numpy,
                                             test_pred=self.query_pred,
                                             final_cfs=self.final_cfs_encoded[:,np.newaxis,:],
                                             final_cfs_preds=self.cfs_preds,
                                             final_cfs_sparse=self.final_cfs_sparse[:,np.newaxis,:] if self.final_cfs_sparse is not None else self.final_cfs_sparse,
                                             cfs_preds_sparse=self.cfs_preds_sparse,
                                             posthoc_sparsity_param=None,
                                             desired_class=self.desired_class,
                                             num_forward_pass=self.model.num_forward_pass)


class ParallelLocalSearch(BaseSearch):

    def __init__(self, data_interface, model_interface, evaluator):
        super().__init__(data_interface, model_interface, evaluator)  # initiating data related parameters

    def generate_counterfactuals(self, query_instance, verbose=False, **kwargs):
        """Generate counterfactuals by randomly sampling features."""
        self.stopping_threshold = 0.5
        self.query = query_instance
        self.verbose = verbose
        self.random_seed = kwargs['seed']
        self.total_CFs = kwargs['num_cfs']
        # np.random.seed(self.random_seed)
        self.permitted_range = kwargs['permitted_range'] if 'permitted_range' in kwargs.keys() else None
        self.features_to_vary = kwargs['features_to_vary'] if 'features_to_vary' in kwargs.keys() else 'all'
        self.desired_class = kwargs['desired_class'] if 'desired_class' in kwargs.keys() else 'opposite'
        num_parallel_runs = kwargs['num_parallel_runs'] if 'num_parallel_runs' in kwargs.keys() else 5
        perturb_type = kwargs['perturb_type'] if 'perturb_type' in kwargs.keys() else 'one'
        init_type = kwargs['init_type'] if 'init_type' in kwargs.keys() else 'random'
        hamming_dist = kwargs['hamming_dist'] if 'hamming_dist' in kwargs.keys() else 2
        self.user_id = kwargs['user_id']
        self.thresh = kwargs['thresh']
        self.budget = kwargs['budget']
        self.iter_type = kwargs['iter_type']
        self.eval = kwargs['eval']

        self.local_budget = self.budget / num_parallel_runs

        self.set_ranges_and_features_to_vary()
        self.precisions = self.data_interface.get_decimal_precisions(output_type="dict")

        # Do predictions once on the query_instance and reuse across to reduce the number inferences.
        self.setup_query_params()

        start_time = timeit.default_timer()
        cfs_df, score, metrics, self.cfs_pred_scores, self.cfs_pred_classes, validity = self.parallel_local_search(num_parallel_runs, perturb_type, init_type, hamming_dist)
        self.total_cfs_found = sum(validity)
        self.post_process(cfs_df)

        self.elapsed = timeit.default_timer() - start_time
        self.print_end_stats()
        return cf_exp.CounterfactualExamples(data_interface=self.data_interface,
                                             test_instance=self.query_numpy,
                                             test_pred=self.query_pred,
                                             final_cfs=self.final_cfs_encoded[:,np.newaxis,:],
                                             final_cfs_preds=self.cfs_preds,
                                             final_cfs_sparse=self.final_cfs_sparse[:,np.newaxis,:] if self.final_cfs_sparse is not None else self.final_cfs_sparse,
                                             cfs_preds_sparse=self.cfs_preds_sparse,
                                             posthoc_sparsity_param=None,
                                             desired_class=self.desired_class,
                                             num_forward_pass=self.model.num_forward_pass)


class RandomSparseSearch(BaseSearch):

    def __init__(self, data_interface, model_interface, evaluator):
        super().__init__(data_interface, model_interface, evaluator)  # initiating data related parameters

    def generate_counterfactuals(self, query_instance, verbose=False, **kwargs):

        """Generate counterfactuals by randomly sampling features."""
        self.stopping_threshold = 0.5
        self.query = query_instance
        self.verbose = verbose
        self.random_seed = kwargs['seed']
        self.total_CFs = kwargs['num_cfs']
        # np.random.seed(self.random_seed)
        self.permitted_range = kwargs['permitted_range'] if 'permitted_range' in kwargs.keys() else None
        self.features_to_vary = kwargs['features_to_vary'] if 'features_to_vary' in kwargs.keys() else 'all'
        self.desired_class = kwargs['desired_class'] if 'desired_class' in kwargs.keys() else 'opposite'
        posthoc_sparsity_param = kwargs['posthoc_sparsity_param'] if 'posthoc_sparsity_param' in kwargs.keys() else 0.1
        posthoc_sparsity_algorithm = kwargs['posthoc_sparsity_algorithm'] if 'posthoc_sparsity_algorithm' in kwargs.keys() else 'linear'
        self.user_id = kwargs['user_id']
        self.budget = kwargs['budget']

        self.set_ranges_and_features_to_vary()
        self.precisions = self.data_interface.get_decimal_precisions(output_type="dict")
        self.setup_query_params()

        start_time = timeit.default_timer()

        leftover_budget = self.budget - self.model.num_forward_pass
        max_feat_to_vary = len(self.features_to_vary) #int(len(self.features_to_vary)/2)+1
        fracs = [1/(1.2**a) for a in np.arange(max_feat_to_vary-1)]
        fracs_norm = fracs / sum(fracs)
        sample_size_list = [round(leftover_budget * f) for f in fracs_norm]
        # sample_size_list = [int(leftover_budget/max_feat_to_vary)] * max_feat_to_vary

        candidate_cfs = None
        # Loop to change one feature at a time, then two features, and so on.
        for num_features_to_vary in range(1, max_feat_to_vary):
            tmp_cfs = self.initialise_cfs(f'ham_{num_features_to_vary}', sample_size_list[num_features_to_vary-1]*3)
            tmp_cfs.drop_duplicates(inplace=True)
            candidate_cfs = pd.concat([candidate_cfs, tmp_cfs])
            if len(candidate_cfs) > leftover_budget:
                break
        candidate_cfs = candidate_cfs.head(leftover_budget)
        score, metric, tmp_pred_scores, tmp_pred_classes, tmp_pred_validity = self.get_score_and_metrics(candidate_cfs, return_preds=True)
        tmp_pred_scores, metric = tmp_pred_scores[tmp_pred_validity], metric[tmp_pred_validity]
        candidate_cfs = candidate_cfs.loc[tmp_pred_validity]
        candidate_cfs.reset_index(inplace=True, drop=True)
        tmp_pred_validity = tmp_pred_validity[tmp_pred_validity]
        if len(metric) <= self.total_CFs:
            best_k_idx = np.arange(len(metric))
        else:
            best_k_idx = np.argpartition(metric, self.total_CFs)[:self.total_CFs]
        cfs_df = candidate_cfs.loc[best_k_idx]
        self.cfs_pred_scores = tmp_pred_scores[best_k_idx]
        self.total_cfs_found = sum(tmp_pred_validity[best_k_idx])
        if self.total_cfs_found == 0:
            cfs_df = self.query_df
            cfs_df = cfs_df.append([cfs_df] * (self.total_CFs-1), ignore_index=True)
            self.cfs_pred_scores = np.repeat(self.query_pred, self.total_CFs)[:, np.newaxis]
        self.post_process(cfs_df)

        # test_instance_df = self.query_df.copy()
        # test_instance_df[self.data_interface.outcome_name] = np.array(np.round(self.query_class, self.outcome_precision))

        # # post-hoc operation on continuous features to enhance sparsity - only for public data
        # if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and \
        #         self.final_cfs is not None and 'data_df' in self.data_interface.__dict__:
        #     query_instance_sparse = self.data_interface.transform_data(test_instance_df.copy()[self.data_interface.feature_names], encode=True, normalise=True, return_numpy=True)
        #     final_cfs_df_sparse = copy.deepcopy(self.final_cfs_encoded)
        #     final_cfs_preds_sparse = copy.deepcopy(self.cfs_pred_scores)
        #     self.final_cfs_sparse, self.cfs_preds_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse, final_cfs_preds_sparse, query_instance_sparse, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        # else:
        #     self.final_cfs_sparse, self.cfs_preds_sparse = None, None

        self.elapsed = timeit.default_timer() - start_time
        self.print_end_stats()

        return cf_exp.CounterfactualExamples(data_interface=self.data_interface,
                                             test_instance=self.query_numpy,
                                             test_pred=self.query_pred,
                                             final_cfs=self.final_cfs_encoded[:,np.newaxis,:],
                                             final_cfs_preds=self.cfs_preds,
                                             final_cfs_sparse=self.final_cfs_sparse[:,np.newaxis,:] if self.final_cfs_sparse is not None else self.final_cfs_sparse,
                                             cfs_preds_sparse=self.cfs_preds_sparse,
                                             posthoc_sparsity_param=None,
                                             desired_class=self.desired_class,
                                             num_forward_pass=self.model.num_forward_pass)


