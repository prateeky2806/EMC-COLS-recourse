"""Module containing all required information about the raw or transformed public data_loaders."""

import pandas as pd
import numpy as np
from scipy import stats
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import logging
from collections import defaultdict

import tensorflow as tf


class PandasDataLoader:
    """A data_loaders interface for public data_loaders."""

    def __init__(self, params):
        """Init method

        :param dataframe: Pandas DataFrame.
        :param continuous_features: List of names of continuous features. The remaining features are categorical features.
        :param outcome_name: str of outcome feature name.
        :param permitted_range (optional): Dictionary with feature names as keys and permitted range in list as values. Defaults to the range inferred from training data_loaders.
        :param test_size (optional): Proportion of test set split. Defaults to 0.2.
        :param test_split_random_state (optional): Random state for train test split. Defaults to 17.
        :param continuous_features_precision (optional): Dictionary with feature names as keys and precisions as values.
        :param data_name (optional): Dataset name

        """
        # PY: Basic check about the object loss_type for the data, categorical_features and outcome_name.
        self._validate_and_set_outcome_name(params)
        self._validate_and_set_dataframe(params)
        self._validate_and_set_continuous_features(params)
        self.io_obj = params['io_obj']
        self.data_name = self.io_obj.data_name
        self.feature_types = self.io_obj.dataset.feature_types
        self.feature_vals = self.io_obj.dataset.feature_vals
        self.feature_change_restriction = self.io_obj.dataset.feature_change_restriction
        # self.outcome_class_names = self.io_obj.dataset.class_names

        # PY: Setting continuous and categorical feature names and feature indices.
        self.categorical_feature_names = [name for name in self.data_df.columns.tolist() if name not in self.continuous_feature_names+[self.outcome_name]]
        self.feature_names = [name for name in self.data_df.columns.tolist() if name != self.outcome_name]
        self.continuous_feature_indexes = [self.data_df.columns.get_loc(name) for name in self.continuous_feature_names if name in self.data_df]
        self.categorical_feature_indexes = [self.data_df.columns.get_loc(name) for name in self.categorical_feature_names if name in self.data_df]

        # PY: Set continuous feature precision.
        if 'continuous_features_precision' in params:
            self.continuous_features_precision = params['continuous_features_precision']
        else:
            self.continuous_features_precision = None

        # PY: setting categorical variables to category datatype in pandas.
        if len(self.categorical_feature_names) > 0:
            for feature in self.categorical_feature_names:
                self.data_df[feature] = self.data_df[feature].apply(str)
            self.data_df[self.categorical_feature_names] = self.data_df[self.categorical_feature_names].astype('category')

            self.categorical_map_str2liststr = {f: sorted(self.data_df[f].cat.categories.tolist())
                                                for f in self.categorical_feature_names}
            self.categorical_map_int2liststr = {i: sorted(self.data_df[f].cat.categories.tolist())
                                    for i, f in enumerate(self.feature_names) if f in self.categorical_feature_names}

            self.categorical_map_int2str = {}
            self.categorical_map_str2int = {}
            for k, v in self.categorical_map_str2liststr.items():
                self.categorical_map_int2str[k] = {}
                self.categorical_map_str2int[k] = {}

                for idx, cat in enumerate(v):
                    self.categorical_map_int2str[k][idx] = cat
                    self.categorical_map_str2int[k][cat] = idx


        # PY: Checking if the original data_loaders is int or float and based on that converting
        # features to float32 or int32.
        if len(self.continuous_feature_names) > 0:
            for feature in self.continuous_feature_names:
                if self.get_data_type(feature) == 'float':
                    self.data_df[feature] = self.data_df[feature].astype(
                        np.float32)
                else:
                    self.data_df[feature] = self.data_df[feature].astype(
                        np.int32)

        # PY: if categorical variable exists the converts them to dummy/indicator variables using pd.get_dummies().
        if len(self.categorical_feature_names) > 0:
            # self.one_hot_encoded_data = self.one_hot_encode_data(self.data_df)
            self.one_hot_encoded_data = self.transform_data(self.data_df, encode=True, normalise=False, return_numpy=False)
            self.encoded_feature_names = [x for x in self.one_hot_encoded_data.columns.tolist(
            ) if x not in np.array([self.outcome_name])]
        else:
            # one-hot-encoded data_loaders is same as orignial data_loaders if there is no categorical features.
            self.one_hot_encoded_data = self.data_df
            self.encoded_feature_names = self.feature_names

        self.set_train_val_test_splits()

        # PY: Validates and sets permitted ranges.
        self._validate_and_set_permitted_range(params)

        self.percentiles = {}
        for feat in self.feature_names:
            if feat in self.continuous_feature_names:
                feat_range = self.cont_to_range(self.permitted_range[feat])
                self.percentiles[feat] = {i: stats.percentileofscore(self.data_df[feat], i)/100 for i in feat_range}
                self.percentiles[feat] = OrderedDict(sorted(self.percentiles[feat].items()))
                assert np.array_equal(np.array(list(self.percentiles[feat].keys())), feat_range), 'Range order has to be the same.'
            elif feat in self.categorical_feature_names:
                if self.feature_types[feat] == 'ordered':
                    self.percentiles[feat] = OrderedDict()
                    vc = self.data_df[feat].value_counts()
                    ordered_counts = [vc[f] for f in self.feature_vals[feat]]
                    percentile = np.cumsum(ordered_counts) / sum(ordered_counts)
                    for ii, f in enumerate(self.feature_vals[feat]):
                        self.percentiles[feat][f] = percentile[ii]

    def cont_to_range(self, range):
        return np.arange(range[0], range[1]+1).tolist()

    def set_train_val_test_splits(self):
        self.train_df = self.data_df.copy().iloc[self.io_obj.dataset.train_idx].reset_index(drop=True, inplace=False)
        self.val_df = self.data_df.copy().iloc[self.io_obj.dataset.val_idx].reset_index(drop=True, inplace=False)
        self.test_df = self.data_df.copy().iloc[self.io_obj.dataset.test_idx].reset_index(drop=True, inplace=False)
        self.data_name = self.io_obj.data_name

        assert all(self.train_df == self.io_obj.data_train)
        assert all(self.val_df == self.io_obj.data_val)
        assert all(self.test_df == self.io_obj.data_test)
        for f in self.categorical_feature_names:
            assert all(self.train_df[f].cat.categories == self.data_df[f].cat.categories)
            assert all(self.val_df[f].cat.categories == self.data_df[f].cat.categories)
            assert all(self.test_df[f].cat.categories == self.data_df[f].cat.categories)

    def _validate_and_set_permitted_range(self, params):
        """Validate and set the dictionary of permitted ranges for continuous features.
        PY: Checks if the permitted ranges are in the params if yes then set the permitted_range attribute
        and also checks if the features are in the df
        PY: If not provided then it determines the permitted ranges.
        """
        self.permitted_range = None
        if 'permitted_range' in params:
            self.permitted_range = params['permitted_range']
            # PY: checks and set the features range for the rest of the parameters for
            # which the range is not provided.
            if not self.check_features_range():
                raise ValueError(
                    "permitted range of features should be within their original range")
        else:
            self.permitted_range, feature_ranges_orig = self.get_features_range(self.permitted_range)

    def _validate_and_set_dataframe(self, params):
        """Validate and set the dataframe.
        PY: Checks if dataframe argument is passed or not and then sets the class variable data_df.
        PY: Also check if the outcome variable is a column in data_df.
        """
        if 'dataframe' not in params:
            raise ValueError("dataframe not found in params")

        if isinstance(params['dataframe'], pd.DataFrame):
            self.data_df = params['dataframe']
        else:
            raise ValueError("should provide a pandas dataframe")

        if 'outcome_name' in params and params['outcome_name'] not in self.data_df.columns.tolist():
            raise ValueError(
                "outcome_name {0} not found in {1}".format(
                    params['outcome_name'], ','.join(self.data_df.columns.tolist())
                )
            )

    def _validate_and_set_continuous_features(self, params):
        """Validate and set the list of continuous features.
        PY: Check if the continuous features are provided in a list or not.
        Then sets the class variable continuous feature names."""
        if 'continuous_features' not in params:
            raise ValueError('continuous_features should be provided')

        if type(params['continuous_features']) is list:
            self.continuous_feature_names = params['continuous_features']
        else:
            raise ValueError(
                "should provide the name(s) of continuous features in the data_loaders as a list")

    def _validate_and_set_outcome_name(self, params):
        """Validate and set the outcome name.
            PY: Checks if the outcome name exists in the parameters and is a string
            and create a class variable with outcome name."""
        if 'outcome_name' not in params:
            raise ValueError("should provide the name of outcome feature")

        if type(params['outcome_name']) is str:
            self.outcome_name = params['outcome_name']
        else:
            raise ValueError("should provide the name of outcome feature as a string")

    def check_features_range(self):
        """ PY: Check if the given ranges lie in the datarange and if not then return false.
        Also sets the permitted ranges for the other features for which it is not provided"""
        for feature in self.continuous_feature_names:
            if feature in self.permitted_range:
                min_value = self.train_df[feature].min()
                max_value = self.train_df[feature].max()

                if self.permitted_range[feature][0] < min_value and self.permitted_range[feature][1] > max_value:
                    return False
            else:
                self.permitted_range[feature] = [self.train_df[feature].min(), self.train_df[feature].max()]
        return True

    def get_features_range(self, permitted_range_input=None, type='all'):
        ranges, cont_ranges, cat_ranges = {}, {}, {}
        # Getting default ranges based on the dataset
        for feature_name in self.continuous_feature_names:
            ranges[feature_name] = [
                self.data_df[feature_name].min(), self.data_df[feature_name].max()]
        for feature_name in self.categorical_feature_names:
            ranges[feature_name] = sorted(self.data_df[feature_name].unique().tolist())
        feature_ranges_orig = ranges.copy()
        # Overwriting the ranges for a feature if input provided
        if permitted_range_input is not None:
            for feature_name, feature_range in permitted_range_input.items():
                ranges[feature_name] = feature_range
        if type == 'all':
            return ranges, feature_ranges_orig
        elif type == 'cont':
            for feat in self.continuous_feature_names:
                cont_ranges[feat] = ranges[feat]
            return cont_ranges, feature_ranges_orig
        elif type == 'cat':
            for feat in self.categorical_feature_names:
                cat_ranges[feat] = ranges[feat]
            return cat_ranges, feature_ranges_orig

    def get_data_type(self, col):
        """Infers data_loaders loss_type of a feature from the training data_loaders."""
        if((self.data_df[col].dtype == np.int64) or (self.data_df[col].dtype == np.int32)):
            return 'int'
        elif((self.data_df[col].dtype == np.float64) or (self.data_df[col].dtype == np.float32)):
            return 'float'
        else:
            raise ValueError("Unknown data_loaders loss_type of feature %s: must be int or float" %col)

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.train_df[feature_name].max()
            min_value = self.train_df[feature_name].min()
            result[feature_name] = (
                df[feature_name] - min_value) / (max_value - min_value)
        return result

    def de_normalize_data(self, df):
        """De-normalizes continuous features from [0,1] range to original range."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.train_df[feature_name].max()
            min_value = self.train_df[feature_name].min()
            result[feature_name] = (
                df[feature_name]*(max_value - min_value)) + min_value
        return result

    def get_minx_maxx(self, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized form."""
        minx = np.array([[0.0]*len(self.encoded_feature_names)])
        maxx = np.array([[1.0]*len(self.encoded_feature_names)])

        for idx, feature_name in enumerate(self.continuous_feature_names):
            max_value = self.train_df[feature_name].max()
            min_value = self.train_df[feature_name].min()

            if normalized:
                minx[0][idx] = (self.permitted_range[feature_name]
                                [0] - min_value) / (max_value - min_value)
                maxx[0][idx] = (self.permitted_range[feature_name]
                                [1] - min_value) / (max_value - min_value)
            else:
                minx[0][idx] = self.permitted_range[feature_name][0]
                maxx[0][idx] = self.permitted_range[feature_name][1]
        return minx, maxx

    def get_mads(self, normalized=False):
        """Computes Median Absolute Deviation of features."""

        mads = {}
        if normalized is False:
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(self.train_df[feature].values - np.median(self.train_df[feature].values)))
        else:
            normalized_train_df = self.normalize_data(self.train_df)
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(normalized_train_df[feature].values - np.median(normalized_train_df[feature].values)))
        return mads

    def get_valid_mads(self, normalized=False, display_warnings=False, return_mads=True):
        """Computes Median Absolute Deviation of features. If they are <=0, returns a practical value instead"""
        mads = self.get_mads(normalized=normalized)
        for feature in mads:
            if mads[feature] <= 0:
                mads[feature] = 1.0
                if display_warnings:
                    logging.warning(" MAD for feature %s is 0, so replacing it with 1.0 to avoid error.", feature)
        if return_mads:
            return mads

    def get_quantiles_from_training_data(self, quantile=0.05, normalized=False):
        """Computes required quantile of Absolute Deviations of features."""

        quantiles = {}
        if normalized is False:
            for feature in self.continuous_feature_names:
                quantiles[feature] = np.quantile(
                    abs(list(set(self.train_df[feature].tolist())) - np.median(list(set(self.train_df[feature].tolist())))), quantile)
        else:
            normalized_train_df = self.normalize_data(self.train_df)
            for feature in self.continuous_feature_names:
                quantiles[feature] = np.quantile(
                    abs(list(set(normalized_train_df[feature].tolist())) - np.median(list(set(normalized_train_df[feature].tolist())))), quantile)
        return quantiles

    def get_data_params(self):
        """Gets all data_loaders related params for DiCE."""
        # PY: Returns the normalized/unnormalized ranges for each of the features based on their permitted range.
        minx, maxx = self.get_minx_maxx(normalized=True)

        # PY: get the column indexes of categorical features after one-hot-encoding
        # PY: List of list where each sublist corresponds to the column for a particular feature.
        self.encoded_categorical_feature_indexes = self.get_encoded_categorical_feature_indexes()

        return minx, maxx, self.encoded_categorical_feature_indexes

    def get_encoded_categorical_feature_indexes(self):
        """Gets the column indexes categorical features after one-hot-encoding."""
        cols = []
        for col_parent in self.categorical_feature_names:
            temp = [self.encoded_feature_names.index(
                col) for col in self.encoded_feature_names if col.startswith(col_parent) and
                   col not in self.continuous_feature_names]
            cols.append(temp)
        return cols

    def get_indexes_of_features_to_vary(self, features_to_vary='all'):
        """Gets indexes from feature names of one-hot-encoded data_loaders."""
        if features_to_vary == "all":
            return [i for i in range(len(self.encoded_feature_names))]
        else:
            ixs = []
            encoded_cats_ixs = self.get_encoded_categorical_feature_indexes()
            encoded_cats_ixs = [item for sublist in encoded_cats_ixs for item in sublist]
            for colidx, col in enumerate(self.encoded_feature_names):
                if colidx in encoded_cats_ixs and col.startswith(tuple(features_to_vary)):
                    ixs.append(colidx)
                elif colidx not in encoded_cats_ixs and col in features_to_vary:
                    ixs.append(colidx)
            return ixs

    def decode_intdf_to_catdf(self, data):
        temp_data = data.copy()
        temp_data = temp_data.replace(self.categorical_map_int2str)
        temp_data[self.categorical_feature_names] = temp_data[
            self.categorical_feature_names].astype('category')
        return temp_data

    def encode_catdf_to_intdf(self, data):
        temp_data = data.copy()
        temp_data = temp_data.replace(self.categorical_map_str2int)
        return temp_data

    def from_dummies(self, data, prefix_sep='_'):
        """Gets the original data_loaders from dummy encoded data_loaders with k levels."""
        out = data.copy()
        if 'binary' in self.data_name:
            out = self.decode_intdf_to_catdf(out)
        else:
            for feat in self.categorical_feature_names:
                # first, derive column names in the one-hot-encoded data_loaders from the original data_loaders
                cat_col_values = []
                for val in list(self.data_df[feat].unique()):
                    cat_col_values.append(feat + prefix_sep + str(val)) # join original feature name and its unique values , ex: education_school
                match_cols = [c for c in data.columns if c in cat_col_values] # check for the above matching columns in the encoded data_loaders

                # then, recreate original data_loaders by removing the suffixes - based on the GitHub issue comment: https://github.com/pandas-dev/pandas/issues/8745#issuecomment-417861271
                cols, labs = [[c.replace(
                    x, "") for c in match_cols] for x in ["", feat + prefix_sep]]
                out[feat] = pd.Categorical(
                    np.array(labs)[np.argmax(data[cols].values, axis=1)])
                out.drop(cols, axis=1, inplace=True)
        return out

    def get_decimal_precisions(self, output_type="list"):
        """"Gets the precision of continuous features in the data."""
        # if the precision of a continuous feature is not given, we use the maximum precision of the modes to capture the precision of majority of values in the column.
        precisions_dict = defaultdict(int)
        precisions = [0] * len(self.feature_names)
        for ix, col in enumerate(self.continuous_feature_names):
            if ((self.continuous_features_precision is not None) and (col in self.continuous_features_precision)):
                precisions[ix] = self.continuous_features_precision[col]
                precisions_dict[col] = self.continuous_features_precision[col]
            elif ((self.data_df[col].dtype == np.float32) or (self.data_df[col].dtype == np.float64)):
                modes = self.data_df[col].mode()
                maxp = len(str(modes[0]).split('.')[1])  # maxp stores the maximum precision of the modes
                for mx in range(len(modes)):
                    prec = len(str(modes[mx]).split('.')[1])
                    if prec > maxp:
                        maxp = prec
                precisions[ix] = maxp
                precisions_dict[col] = maxp
        if output_type == "list":
            return precisions
        elif output_type == "dict":
            return precisions_dict

    def get_decoded_data(self, data):
        """Gets the original data_loaders from dummy encoded data_loaders."""
        if isinstance(data, np.ndarray):
            index = [i for i in range(0, len(data))]
            data = pd.DataFrame(data=data, index=index,
                                columns=self.encoded_feature_names)
        return self.from_dummies(data)

    def prepare_df_for_encoding(self):
        """Facilitates prepare_query_instance() function.
        PY: Creates a df with columns as categorical features and rows as the unique values they can take."""
        levels = []
        colnames = self.categorical_feature_names
        for cat_feature in colnames:
            levels.append(sorted(self.data_df[cat_feature].cat.categories.tolist()))

        if len(colnames) > 0:
            df = pd.DataFrame({colnames[0]: levels[0]})
        else:
            df = pd.DataFrame()

        for col in range(1, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: levels[col]})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        colnames = self.continuous_feature_names
        for col in range(0, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: []})
            df = pd.concat([df, temp_df], axis=1, sort=False)
        return df

    def prepare_query_instance(self, query_instance, encode):
        """Prepares user defined test input for DiCE."""

        if isinstance(query_instance, list):
            query_instance = {'row1': query_instance}
            test = pd.DataFrame.from_dict(
                query_instance, orient='index', columns=self.feature_names)

        elif isinstance(query_instance, dict):
            query_instance = dict(zip(query_instance.keys(), [[q] for q in query_instance.values()]))
            test = pd.DataFrame(query_instance, columns=self.feature_names)
        else:
            raise ValueError("Query instance can either be a list or a dict.")

        test = test.reset_index(drop=True)
        return self.transform_data(test, encode, normalise=True, return_numpy=False)

    def transform_data(self, data, encode, normalise, return_numpy=False):
        """" Written by XXXXXXX XXXXX
        If encode is False, then normalise continuous variables
        If encode is True, ohe data and normalise it."""
        if encode:
            # PY: Creates a df with columns as categorical features and rows as the unique values they can take.
            # Appending all classes because ohe has to be wrt to all the categories.
            if 'binary' in self.data_name:
                ret = self.encode_catdf_to_intdf(data)
                if self.outcome_name in data.columns:
                    features = self.continuous_feature_names + self.categorical_feature_names + [self.outcome_name]
                else:
                    features = self.continuous_feature_names + self.categorical_feature_names
                ret = ret[features]
            else:
                temp = self.prepare_df_for_encoding()
                temp = temp.append(data, ignore_index=True, sort=False)
                temp = pd.get_dummies(temp, drop_first=False, columns=self.categorical_feature_names)
                ret = temp.tail(data.shape[0]).reset_index(drop=True)
            if normalise:
                ret = self.normalize_data(ret)
        elif encode is False and normalise is True:
            ret = self.normalize_data(data)
        else:
            raise ValueError("Atleast on of encode or normalise has to be True")
        if return_numpy:
            return ret.values.astype(np.float)
        else:
            return ret

    def get_dev_data(self, model_interface, desired_class, filter_threshold=0.5):
        """Constructs dev data_loaders by extracting part of the test data_loaders for which finding counterfactuals make sense."""

        # create TensorFLow session if one is not already created
        if tf.get_default_session() is not None:
            self.data_sess = tf.get_default_session()
        else:
            self.data_sess = tf.InteractiveSession()

        # loading trained model_loaders
        model_interface.load_model()

        # get the permitted range of change for each feature
        minx, maxx = self.get_minx_maxx(normalized=True)

        # get the transformed data_loaders: continuous features are normalized to fall in the range [0,1], and categorical features are one-hot encoded
        data_df_transformed = self.normalize_data(self.one_hot_encoded_data)

        # split data_loaders - nomralization considers only train df and there is no leakage due to transformation before train-test splitting
        _, test = self.split_data(data_df_transformed)
        test = test.drop_duplicates(
            subset=self.encoded_feature_names).reset_index(drop=True)

        # finding target predicted probabilities
        input_tensor = tf.Variable(minx, dtype=tf.float32)
        output_tensor = model_interface.get_output(
            input_tensor)  # model_loaders(input_tensor)
        temp_data = test[self.encoded_feature_names].values.astype(np.float32)
        dev_preds = [self.data_sess.run(output_tensor, feed_dict={
                                        input_tensor: np.array([dt])}) for dt in temp_data]
        dev_preds = [dev_preds[i][0][0] for i in range(len(dev_preds))]

        # filtering examples which have predicted value >/< threshold
        dev_data = test[self.encoded_feature_names]
        if desired_class == 0:
            idxs = [i for i in range(len(dev_preds))
                    if dev_preds[i] > filter_threshold]
        else:
            idxs = [i for i in range(len(dev_preds))
                    if dev_preds[i] < filter_threshold]
        dev_data = dev_data.iloc[idxs]
        dev_preds = [dev_preds[i] for i in idxs]

        # convert from one-hot encoded vals to user interpretable fromat
        dev_data = self.from_dummies(dev_data)
        dev_data = self.de_normalize_data(dev_data)
        return dev_data[self.feature_names], dev_preds  # values.tolist()
