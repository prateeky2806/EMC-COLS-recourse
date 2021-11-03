"""Module containing a template class to generate counterfactual explanations.
   Subclasses implement interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All methods are in dice.explainers"""

import numpy as np

class ExplainerBase:

    def __init__(self, data_interface):
        """Init method

        :param data_interface: an interface class to access data_loaders related params.
        """
        # get data_loaders-related parameters - minx and max for normalized continuous features
        self.data_interface = data_interface
        self.minx, self.maxx, self.encoded_categorical_feature_indexes = self.data_interface.get_data_params()

        # min and max for continuous features in original scale
        flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
        self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]
        org_minx, org_maxx = self.data_interface.get_minx_maxx(normalized=False)
        self.cont_minx = list(org_minx[0][self.encoded_continuous_feature_indexes])
        self.cont_maxx = list(org_maxx[0][self.encoded_continuous_feature_indexes])

        # decimal precisions for continuous features
        self.cont_precisions = [self.data_interface.get_decimal_precisions()[ix] for ix in self.encoded_continuous_feature_indexes]

    def generate_counterfactuals(self):
        raise NotImplementedError

    def do_posthoc_sparsity_enhancement(self, final_cfs_sparse, cfs_preds_sparse, query_instance, posthoc_sparsity_param, posthoc_sparsity_algorithm):
        """Post-hoc method to encourage sparsity in a generated counterfactuals.

        :param final_cfs_sparse: List of final CFs in numpy format.
        :param cfs_preds_sparse: List of predicted outcomes of final CFs in numpy format.
        :param query_instance: Query instance in numpy format.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model_loaders.
        """

        normalized_quantiles = self.data_interface.get_quantiles_from_training_data(quantile=posthoc_sparsity_param, normalized=True)
        normalized_mads = self.data_interface.get_valid_mads(normalized=True)
        for feature in normalized_quantiles:
            normalized_quantiles[feature] = min(normalized_quantiles[feature], normalized_mads[feature])

        features_sorted = sorted(normalized_quantiles.items(), key=lambda kv: kv[1], reverse=True)
        for ix in range(len(features_sorted)):
            features_sorted[ix] = features_sorted[ix][0]
        decimal_prec = self.data_interface.get_decimal_precisions()[0:len(self.encoded_continuous_feature_indexes)]

        for cf_ix in range(self.total_CFs):
            for feature in features_sorted:
                # current_pred = self.predict_fn(final_cfs_sparse[cf_ix])
                current_pred = self.predict_probs(final_cfs_sparse[cf_ix], return_last=True, return_numpy=True)[np.newaxis]
                feat_ix = self.data_interface.encoded_feature_names.index(feature)
                diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]

                if(abs(diff) <= normalized_quantiles[feature]):
                    if posthoc_sparsity_algorithm == "linear":
                        final_cfs_sparse[cf_ix] = do_linear_search(self, diff, decimal_prec, query_instance, cf_ix, feat_ix, final_cfs_sparse, current_pred)

                    elif posthoc_sparsity_algorithm == "binary":
                        final_cfs_sparse[cf_ix] = do_binary_search(self, diff, decimal_prec, query_instance, cf_ix, feat_ix, final_cfs_sparse, current_pred)

        # cfs_preds_sparse[cf_ix] = self.predict_fn(final_cfs_sparse[cf_ix])
        cfs_preds_sparse = self.predict_probs(np.array(final_cfs_sparse), return_last=True, return_numpy=True)

        return final_cfs_sparse, cfs_preds_sparse

    def infer_target_cfs_class(self, desired_class_input, original_pred, num_output_nodes):
        """ Infer the target class for generating CFs. Only called when
            model_type=="classifier".
            TODO: Add support for opposite desired class in multiclass. Downstream methods should decide
                  whether it is allowed or not.
        """
        if desired_class_input == "opposite":
            if num_output_nodes == 2:
                original_pred_1 = 1 if original_pred > 0.5 else 0
                target_class = int(1 - original_pred_1)
                return target_class
            elif num_output_nodes > 2:
                raise ValueError(
                    "Desired class cannot be opposite if the number of classes is more than 2.")
        elif isinstance(desired_class_input, int):
            if desired_class_input >= 0 and desired_class_input < num_output_nodes:
                target_class = desired_class_input
                return target_class
            else:
                raise ValueError("Desired class not present in training data!")
        else:
            raise ValueError("The target class for {0} could not be identified".format(
                                                desired_class_input))

    def get_model_output_from_scores(self, model_scores):
        model_output = np.zeros(len(model_scores), dtype=np.int32)
        for i in range(len(model_scores)):
            model_output[i] = 1 if model_scores[i] >= 0.5 else 0
        return model_output

    def decide_cf_validity(self, model_outputs):
        validity = np.zeros(len(model_outputs), dtype=np.int32)
        for i in range(len(model_outputs)):
            pred = model_outputs[i]
            if self.num_output_nodes == 2: # binary
                pred_1 = pred[-1]
                validity[i] = 1 if ((self.target_cf_class == 0 and pred_1 < self.stopping_threshold) or (self.target_cf_class == 1 and pred_1>= self.stopping_threshold)) else 0
            else:
                raise ValueError(f"Only works for 2 classes.")# multiclass
        return validity


def do_linear_search(self, diff, decimal_prec, query_instance, cf_ix, feat_ix, final_cfs_sparse, current_pred):
    """Performs a greedy linear search - moves the continuous features in CFs towards original values in query_instance greedily until the prediction class changes."""

    old_diff = diff
    it = 0 # To break out of infinite loop if the model gets stuck in the while loop.
    try:
        change = (10**-decimal_prec[feat_ix])/(self.cont_maxx[feat_ix] - self.cont_minx[feat_ix]) # the minimal possible change for a feature
    except:
        pass
    while((abs(diff)>10e-4) and (np.sign(diff*old_diff) > 0) and
          ((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or
           (self.target_cf_class == 1 and current_pred > self.stopping_threshold)) and it<10000): # move until the prediction class changes
        old_val = final_cfs_sparse[cf_ix].ravel()[feat_ix]
        final_cfs_sparse[cf_ix].ravel()[feat_ix] += np.sign(diff)*change
        # current_pred = self.predict_fn(final_cfs_sparse[cf_ix])
        current_pred = self.predict_probs(final_cfs_sparse[cf_ix])[np.newaxis]
        old_diff = diff
        it += 1
        if(((self.target_cf_class == 0 and current_pred > self.stopping_threshold) or (self.target_cf_class == 1 and current_pred < self.stopping_threshold))):
            final_cfs_sparse[cf_ix].ravel()[feat_ix] = old_val
            diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]
            return final_cfs_sparse[cf_ix]

        diff = query_instance.ravel()[feat_ix] - final_cfs_sparse[cf_ix].ravel()[feat_ix]

    return final_cfs_sparse[cf_ix]

def do_binary_search(self, diff, decimal_prec, query_instance, cf_ix, feat_ix, final_cfs_sparse, current_pred):
    """Performs a binary search between continuous features of a CF and corresponding values in query_instance until the prediction class changes."""

    old_val = final_cfs_sparse[cf_ix].ravel()[feat_ix]
    final_cfs_sparse[cf_ix].ravel()[feat_ix] = query_instance.ravel()[feat_ix]
    current_pred = self.predict_fn(final_cfs_sparse[cf_ix])

    # first check if assigning query_instance values to a CF is required.
    if(((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
        return final_cfs_sparse[cf_ix]
    else:
        final_cfs_sparse[cf_ix].ravel()[feat_ix] = old_val

    # move the CF values towards the query_instance
    if diff > 0:
        left = final_cfs_sparse[cf_ix].ravel()[feat_ix]
        right = query_instance.ravel()[feat_ix]

        while left <= right:
            current_val = left + ((right - left)/2)
            current_val = round(current_val, decimal_prec[feat_ix])

            final_cfs_sparse[cf_ix].ravel()[feat_ix] = current_val
            current_pred = self.predict_fn(final_cfs_sparse[cf_ix])

            if current_val == right or current_val == left:
                break

            if(((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
                left = current_val + (10**-decimal_prec[feat_ix])
            else:
                right = current_val - (10**-decimal_prec[feat_ix])

    else:
        left = query_instance.ravel()[feat_ix]
        right = final_cfs_sparse[cf_ix].ravel()[feat_ix]

        while right >= left:
            current_val = right - ((right - left)/2)
            current_val = round(current_val, decimal_prec[feat_ix])

            final_cfs_sparse[cf_ix].ravel()[feat_ix] = current_val
            current_pred = self.predict_fn(final_cfs_sparse[cf_ix])

            if current_val == right or current_val == left:
                break

            if(((self.target_cf_class == 0 and current_pred < self.stopping_threshold) or (self.target_cf_class == 1 and current_pred > self.stopping_threshold))):
                right = current_val - (10**-decimal_prec[feat_ix])
            else:
                left = current_val + (10**-decimal_prec[feat_ix])

    return final_cfs_sparse[cf_ix]
