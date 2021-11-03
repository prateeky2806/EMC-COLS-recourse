import numpy as np
from tqdm import tqdm


class MetricsEvaluator():
    def __init__(self, data_loader):
        self.dataclass = data_loader
        self.selected_metrics = ['diversity', 'proximity', 'validity']
        self.minimize_metric = [-1, -1, -1]
        self.dirichlet_weights = np.random.dirichlet(np.ones(len(self.selected_metrics)), size=3000)
        self.max = 99999

    """Computes all the metrics for generated counterfactuals."""
    def get_metrics(self, cfs_list, queries, original_class):
        """cfs_list, last element is the class"""
        pred_classes = []
        for cfs in cfs_list:
            pred_class_cf = []
            for cf in cfs:
                pred_class_cf.append(cf.pop())
            pred_classes.append(np.array(pred_class_cf))

        self.gen_pred_class = pred_classes
        self.generations = cfs_list
        self.queries = queries
        self.original_class = original_class
        self.not_empty = []
        self.metrics = []
        self.metric_dict = []
        self.scores = []
        for ix, gen in enumerate(self.generations):
            query = np.array(list(self.queries[ix].values()))
            gen = np.array(gen)
            pc = np.array(self.gen_pred_class[ix])

            sample_metrics, metrics_name = self.get_single_sample_metrics(gen, query, self.original_class[ix], pc, return_dict=True)
            score = self.get_DS(sample_metrics)
            sample_metrics['metric_score'] = score
            metrics_name += ['metric_score']
            self.metric_dict.append(sample_metrics)
            self.metrics.append(list(sample_metrics.values()))
            self.scores.append(score)
            self.not_empty.append(0 if ((len(gen) == 0) or (self.max in sample_metrics.values())) else 1)

        self.metrics_name = metrics_name
        self.avg_metrics = np.array(self.metrics)[np.array(self.not_empty).astype(bool)].mean(0)
        self.avg_metric_dict = {n: round(self.avg_metrics[i], 4) for i, n in enumerate(self.metrics_name)}
        self.avg_score = np.array(self.scores)[np.array(self.not_empty).astype(bool)].mean()
        print('Metric Computed Successfully')
        print(self.avg_metric_dict)
        print(f'Average Score: {self.avg_score:.6f}')

    def get_single_sample_metrics(self, cfs, query, original_class, pred_classes, type='all', return_dict=False):
        if type == 'all' or type == 'dirichlet':
            sample_validity, names_val, unique_valid_cfs = self.get_validity(cfs, original_class, pred_classes)
            sample_proximity, names_prox = self.get_proximity(unique_valid_cfs, query)
            sample_diversity, names_div = self.get_diversity(unique_valid_cfs)
            sample_sparsity, names_spars = self.get_sparsity(unique_valid_cfs, query)
            sample_metrics = list(sample_validity + sample_proximity + sample_diversity + sample_sparsity)
            metrics_name = names_val + names_prox + names_div + names_spars
            prox = self.merge_cat_cont(sample_proximity[0], sample_proximity[2])
            div = self.merge_cat_cont(sample_diversity[0], sample_diversity[2])
            sample_metrics += [prox, div]
            metrics_name += ['proximity', 'diversity']

        elif type == 'diversity':
            sample_validity, names_val, unique_valid_cfs = self.get_validity(cfs, original_class, pred_classes)
            sample_diversity, names_div = self.get_diversity(unique_valid_cfs)
            sample_metrics = list(sample_validity + sample_diversity)
            metrics_name = names_val + names_div
            div = self.merge_cat_cont(sample_diversity[0], sample_diversity[2])
            sample_metrics += [div]
            metrics_name += ['diversity']

        elif type == 'proximity':
            sample_validity, names_val, unique_valid_cfs = self.get_validity(cfs, original_class, pred_classes)
            sample_proximity, names_prox = self.get_proximity(unique_valid_cfs, query)
            sample_metrics = list(sample_validity + sample_proximity)
            metrics_name = names_val + names_prox
            prox = self.merge_cat_cont(sample_proximity[0], sample_proximity[2])
            sample_metrics += [prox]
            metrics_name += ['proximity']

        elif type == 'sparsity':
            sample_validity, names_val, unique_valid_cfs = self.get_validity(cfs, original_class, pred_classes)
            sample_sparsity, names_spars = self.get_sparsity(unique_valid_cfs, query)
            sample_metrics = list(sample_validity + sample_sparsity)
            metrics_name = names_val + names_spars
        else:
            raise ValueError('Invalid value.')

        if return_dict:
            return {metrics_name[i]: sample_metrics[i] for i in range(len(metrics_name))}, metrics_name
        else:
            return sample_metrics, metrics_name

    def get_DS(self, metrics):
        metric_vals = [metrics[met] for met in self.selected_metrics]
        return np.dot(self.dirichlet_weights, np.array(metric_vals)).mean(0)

    def get_score_from_metric(self, metrics, type='dirichlet'):
        if type == 'diversity':
            return (-1 * metrics['diversity'] - metrics['validity']) / 2
        elif type == 'proximity':
            return (-1 * metrics['proximity'] - metrics['validity']) / 2
        elif type == 'sparsity':
            return (-1 * metrics['sparsity'] - metrics['validity']) / 2
        elif type == 'dirichlet':
            return -1 * self.get_DS(metrics)

    def merge_cat_cont(self, cat, cont):
        num_cont = len(self.dataclass.continuous_feature_names)
        num_cat = len(self.dataclass.categorical_feature_names)
        weights_cat_cont = np.array([num_cat, num_cont])
        weights_cat_cont = weights_cat_cont / weights_cat_cont.sum(0)
        return np.dot(weights_cat_cont, np.array([cat, cont]))

    def get_distance(self, cfs, query):
        """Computes distance between (2d-array, 1d-array) or (1d-array, 1d-array)
        Considers only the cat and cont feature indices, ignores everything else.
        For continuous features the distance it is assumed that the variables are integers.
        outputs distance in order cat_count_dists, cont_dists, cont_count_dists.
        """
        cfs = np.array(cfs)
        query = np.array(query)

        cat_idx = self.dataclass.categorical_feature_indexes
        cont_idx = self.dataclass.continuous_feature_indexes
        cont_mads_dict = self.dataclass.get_mads()
        cont_feat_range, _ = self.dataclass.get_features_range(None, 'cont')

        norm_const, cont_mads = [], []
        for feat in self.dataclass.continuous_feature_names:
            cont_mads.append(cont_mads_dict[feat] if cont_mads_dict[feat] > 0 else 1)
            r = cont_feat_range[feat][1] - cont_feat_range[feat][0]
            norm_const.append(r if r > 0 else 0)
        cont_dists = {}

        cat_count_dists = (cfs[..., cat_idx] != query[..., cat_idx]).astype(int).mean(-1)
        cont_count_dists = (cfs[..., cont_idx].astype(float).astype(int) != query[..., cont_idx].astype(float).astype(int)).astype(int).mean(-1)
        # cont_diff = (np.abs(cfs[..., cont_idx].astype(float).astype(int) - query[..., cont_idx].astype(float).astype(int)))

        cont_dists_mad = ((np.abs(cfs[..., cont_idx].astype(float).astype(int) - query[..., cont_idx].astype(float).astype(int)))/cont_mads).mean(-1)
        cont_dists_norm = ((np.abs(cfs[..., cont_idx].astype(float).astype(int) - query[..., cont_idx].astype(float).astype(int)))/norm_const).mean(-1)

        return [cat_count_dists, [cont_dists_mad, cont_dists_norm], cont_count_dists]

    def get_validity(self, cfs, original_class, pred_classes):
        if len(cfs) == 0:
            return tuple([0]), ['validity'], np.array([], dtype=np.int)

        target_class = 1 - int(original_class)
        unique_cfs, unique_idx = np.unique(cfs, axis=0, return_index=True)
        pred_classes = np.array(pred_classes)
        validity_mask = ((pred_classes[unique_idx]).astype(int) == target_class)
        validity_mask_all = ((pred_classes).astype(int) == target_class)
        valid_perc = (validity_mask.sum()/len(cfs))
        return tuple([valid_perc]), ['validity'], unique_cfs[validity_mask]

    def get_proximity(self, cfs, query):
        """Gives out continuous and categorical proximity metrics
        """
        if len(cfs) == 0:
            return (self.max, self.max, self.max), ['prox_cat', 'prox_cont_mad', 'prox_cont_norm']

        cat_count_dists, cont_dists, cont_count_dists = self.get_distance(cfs, query)
        cat_prox = 1 - cat_count_dists.mean(0)
        cont_prox_mad = -1 * cont_dists[0].mean(0)
        cont_prox_norm = 1 - cont_dists[1].mean(0)
        return (cat_prox, cont_prox_mad, cont_prox_norm), ['prox_cat', 'prox_cont_mad', 'prox_cont_norm']

    def get_diversity(self, cfs):
        if len(cfs) == 1 or len(cfs) == 0:
            return (0, 0, 0, 0), ['diversity_cat', 'diversity_cont_mad', 'diversity_cont_norm', 'diversity_cont_count']
        diversity_pairs = []
        for ix in range(len(cfs)):
            for jx in range(ix+1, len(cfs)):
                diversity_pairs.append(self.get_distance(cfs[ix], cfs[jx]))
        cat_div, cont_div, cont_count_div = zip(*diversity_pairs)
        cat_div = np.array(cat_div).mean(0)
        cont_div_mad, cont_div_norm = zip(*cont_div)
        cont_div_mad = np.array(cont_div_mad).mean(0)
        cont_div_norm = np.array(cont_div_norm).mean(0)
        cont_count_div = np.array(cont_count_div).mean(0)
        return (cat_div, cont_div_mad, cont_div_norm, cont_count_div), ['diversity_cat', 'diversity_cont_mad', 'diversity_cont_norm', 'diversity_cont_count']

    def get_sparsity(self, cfs, query):
        if len(cfs) == 0:
            return (0, 0, 0), ['sparsity_cat', 'sparsity_cont', 'sparsity']

        num_cat = len(self.dataclass.categorical_feature_indexes)
        num_cont = len(self.dataclass.continuous_feature_indexes)

        cat_count_dists, cont_dists, cont_count_dists = self.get_distance(cfs, query)
        sample_spars = (cont_count_dists * num_cont + cat_count_dists * num_cat)/(num_cont + num_cat)
        cat_spars = 1 - cat_count_dists.mean(0)
        cont_spars = 1 - cont_count_dists.mean(0)
        spars = 1 - sample_spars.mean(0)
        return (cat_spars, cont_spars, spars), ['sparsity_cat', 'sparsity_cont', 'sparsity']


class UserEvaluator():
    def __init__(self, data_loader, users_info, score_type='min'):
        self.invalid = 99999  # same as self.max_cost
        self.dataclass = data_loader
        self.users = users_info['users']
        self.cost_map = users_info['cost_map']
        self.score_type = score_type

    def get_metrics(self, user_queries, cfs_list, original_class, thresh, test_alpha=None):
        pred_classes = []
        for cfs in cfs_list:
            pred_class_cf = []
            for cf in cfs:
                pred_class_cf.append(cf.pop())
            pred_classes.append(pred_class_cf)

        self.gen_pred_class = pred_classes
        self.generations = cfs_list
        self.original_class = original_class
        self.user_queries = user_queries
        self.metrics = []
        self.costs, self.user_score, self.satisfied, self.covered, self.validity_masks = [], [], [], [], []
        for user_id, gen in enumerate(self.generations):
            assert np.array_equal(np.array(list(self.users[user_id]['user_cost']['query'].values())), user_queries.loc[user_id].values.astype('<U21'))
            gen = np.array(gen)
            pc = np.array(self.gen_pred_class[user_id])

            cost, validity_mask = self.get_single_sample_cost(gen, user_id, self.original_class[user_id], pc, test_alpha=test_alpha)
            if sum(validity_mask) == 0 or len(gen) == 0:
                user_final_cost = self.invalid
            else:
                user_final_cost = cost[validity_mask].min()

            # Hacky way, its unlikely cost can be higer than 100, if numfeatures is much less than 100.
            user_covered = True if user_final_cost < 100 else False
            user_satisfaction = [True if user_final_cost <= t else False for t in thresh]

            self.costs.append(cost)
            self.validity_masks.append(validity_mask)
            self.user_score.append(user_final_cost)
            self.satisfied.append(user_satisfaction)
            self.covered.append(user_covered)

        self.pop_avg_score = np.array(self.user_score)[self.covered].mean(0)
        self.pop_frac_satisfied = np.array(self.satisfied).mean(0)
        self.pop_coverage = np.array(self.covered).mean()
        print(f'\nTesting at alpha: {test_alpha}\n')
        print(f'Each Users score:\n {self.user_score}')
        print(f'Population average score: {self.pop_avg_score:.4f}')
        print(f'Coverage: {self.pop_coverage:.4f}')
        for i in range(len(thresh)):
            print(f'Fraction Satisfied @ {thresh[i]}: {self.pop_frac_satisfied[i]:.4f}')

    def get_ood_metrics(self, user_queries, cfs_list, original_class, ood_data, thresh):
        pred_classes = []
        for cfs in cfs_list:
            pred_class_cf = []
            for cf in cfs:
                pred_class_cf.append(cf.pop())
            pred_classes.append(pred_class_cf)

        self.gen_pred_class = pred_classes
        self.generations = cfs_list
        self.original_class = original_class
        self.user_queries = user_queries
        self.metrics = []

        num_ood_costs  = max([int(k.split('_')[-1]) for k in ood_data['users'][0].keys() if 'cost' in k]) + 1

        user_concs = []
        for user in self.users:
            conc = []
            for mc in user['metric_costs']:
                conc.append(mc['concentration'])
            user_concs.append(np.array(conc))

        covered_all, satisfied_all, distance_all = [], [], []
        for ood_cost_idx in tqdm(range(num_ood_costs)):
            satisfied = []
            covered = []
            distance = []
            for user_id, gen in enumerate(self.generations):
                assert np.array_equal(np.array(list(self.users[user_id]['user_cost']['query'].values())), user_queries.loc[user_id].values.astype('<U21'))
                gen = np.array(gen)
                pc = np.array(self.gen_pred_class[user_id])
                ood_conc = ood_data['users'][user_id]['concentration'][ood_cost_idx]
                mcmc_conc = user_concs[user_id]
                mcmc_conc[mcmc_conc==1e-7] = 0
                dist = np.linalg.norm(mcmc_conc/mcmc_conc.sum(1, keepdims=True) - ood_conc/ood_conc.sum(keepdims=True), ord=2, axis=1).min()

                # dist = np.linalg.norm(mcmc_conc/mcmc_conc.sum(1, keepdims=True) - ood_conc/ood_conc.sum(keepdims=True), ord=2, axis=1).mean()

                # mcmc_conc_mean = mcmc_conc.mean(0)
                # dist = np.linalg.norm(mcmc_conc_mean/mcmc_conc_mean.sum(keepdims=True) - ood_conc/ood_conc.sum(keepdims=True), ord=2)

                cost, validity_mask = self.get_ood_cost(gen, self.original_class[user_id], pc, ood_data['users'][user_id][f'ood_cost_{ood_cost_idx}'])
                if sum(validity_mask) == 0 or len(gen) == 0:
                    user_final_cost = self.invalid
                else:
                    user_final_cost = cost[validity_mask].min()
                # Hacky way, its unlikely cost can be higer than 100, if numfeatures is much less than 100.
                covered.append(True if user_final_cost < 100 else False)
                satisfied.append([True if user_final_cost <= t else False for t in thresh])
                distance.append(dist)
            covered_all.append(covered)
            satisfied_all.append(satisfied)
            distance_all.append(distance)
        covered_all = np.array(covered_all)
        satisfied_all = np.array(satisfied_all)
        distance_all = np.array(distance_all)
        return covered_all, satisfied_all, distance_all

    def get_ood_cost(self, cfs, original_class, pred_classes, cost_function):
        validity_mask, valid_perc = self.get_validity(cfs, original_class, pred_classes)
        cfs_cost = []
        for ii, cf in enumerate(cfs):
            if validity_mask[ii]:
                mcmc_cost = 0
                for ii, feat in enumerate(self.dataclass.feature_names):
                    if feat in self.dataclass.continuous_feature_names:
                        mcmc_cost += cost_function[feat][self.cost_map[feat][int(float(cf[ii]))]]
                    else:
                        mcmc_cost += cost_function[feat][self.cost_map[feat][cf[ii]]]
            else:
                mcmc_cost = self.invalid
            cfs_cost.append(mcmc_cost)
        return np.array(cfs_cost), validity_mask

    def get_validity(self, cfs, original_class, pred_classes):
        if len(cfs) == 0:
            return np.array([], dtype=np.int), 0

        target_class = 1 - int(original_class)
        pred_classes = np.array(pred_classes)
        validity_mask_all = ((pred_classes).astype(int) == target_class)
        unique_cfs, unique_idx = np.unique(cfs, axis=0, return_index=True)
        validity_mask_unique = ((pred_classes[unique_idx]).astype(int) == target_class)
        valid_perc = (validity_mask_unique.sum()/len(cfs))
        return validity_mask_all, valid_perc

    def get_cf_cost(self, cf, user_info, mcmc, test_alpha):
        if mcmc:
            cf_cost = np.zeros(len(user_info['metric_matrix'][self.dataclass.feature_names[0]]))
        else:
            cf_cost = 0
        for ii, feat in enumerate(self.dataclass.feature_names):
            if feat in self.dataclass.continuous_feature_names:
                if mcmc:
                    cf_cost += user_info['metric_matrix'][feat][...,self.cost_map[feat][int(float(cf[ii]))]]
                else:
                    if test_alpha != None:
                        cf_cost += user_info['user_cost'][f'cost_{test_alpha:.1f}'][feat][self.cost_map[feat][int(float(cf[ii]))]]
                    else:
                        cf_cost += user_info['user_cost'][f'cost'][feat][self.cost_map[feat][int(float(cf[ii]))]]
            else:
                if mcmc:
                    cf_cost += user_info['metric_matrix'][feat][..., self.cost_map[feat][cf[ii]]]
                else:
                    if test_alpha != None:
                        cf_cost += user_info['user_cost'][f'cost_{test_alpha:.1f}'][feat][self.cost_map[feat][cf[ii]]]
                    else:
                        cf_cost += user_info['user_cost'][f'cost'][feat][self.cost_map[feat][cf[ii]]]
        return cf_cost

    def get_single_sample_cost(self, cfs, user_id, original_class, pred_classes, mcmc=False, test_alpha=None):
        user_info = self.users[user_id]
        validity_mask, valid_perc = self.get_validity(cfs, original_class, pred_classes)
        cfs_cost = []
        for ii, cf in enumerate(cfs):
            if validity_mask[ii]:
                mcmc_cost = self.get_cf_cost(cf, user_info, mcmc, test_alpha)
            else:
                if mcmc:
                    mcmc_cost = self.invalid * np.ones(len(user_info['metric_matrix'][self.dataclass.feature_names[0]]))
                else:
                    mcmc_cost = self.invalid
            cfs_cost.append(mcmc_cost)
        return np.array(cfs_cost), validity_mask

    def get_score_from_cost(self, cfs_cost, do_avg):
        if self.score_type == 'min':
            min_cfs_idx_per_mcmc = cfs_cost.argmin(0)
            min_cost_per_mcmc = cfs_cost.min(0)
            if do_avg:
                avg_min_score = min_cost_per_mcmc.mean()
                return avg_min_score
            else:
                return min_cost_per_mcmc, min_cfs_idx_per_mcmc
        else:
            raise NotImplementedError(f'{self.score_type} is not implemented.')