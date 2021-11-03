import copy, os
import numpy as np, pandas as pd
import pickle, argparse
from addict import Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit


def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret


def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})


def load_dataset(data_name, args, balance=True, discretize=True, dataset_folder='./'):
    if data_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_fid = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
        education_map = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
                'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
                'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_map = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
                "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
                "Service", "Priv-house-serv": "Service", "Prof-specialty":
                "Professional", "Protective-serv": "Other", "Sales":
                "Sales", "Tech-support": "Other", "Transport-moving":
                "Blue-Collar",
        }
        country_map = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
                'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
                'British-Commonwealth', 'Iran': 'Other', 'Ireland':
                'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
                'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
                'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
                'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
                'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_map = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
                'Separated', 'Separated': 'Separated', 'Divorced':
                'Separated', 'Widowed': 'Widowed'
        }
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}
        def cap_gains_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

        transformations = {
            3: lambda x: map_array_values(x, education_map),
            5: lambda x: map_array_values(x, married_map),
            6: lambda x: map_array_values(x, occupation_map),
            10: cap_gains_fn,
            11: cap_gains_fn,
            13: lambda x: map_array_values(x, country_map),
            14: lambda x: map_array_values(x, label_map),
        }
        class_map = {'Less than $50,000': 0, 'More than $50,000': 1}
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_fid=categorical_fid, discretize=discretize,
            balance=balance, feature_transformations=transformations, hparams=args, cmap=class_map)
        dataset.class_names = ['Less than $50,000', 'More than $50,000']

        # 0 means can be increased or decreased
        # 1 means can only increase, -1 means can only decrease
        # -2 means it is fixed.
        feature_types = {
            'Age': 'ordered', 'Workclass': 'unordered', 'Education': 'ordered', 'Marital Status': 'unordered',
            'Occupation': 'unordered', 'Relationship': 'unordered', 'Race': 'fixed', 'Sex': 'fixed',
            'Capital Gain': 'ordered', 'Capital Loss': 'ordered', 'Hours per week': 'ordered', 'Country': 'fixed'
        }
        feature_change_restriction = {
            'Age': 1, 'Workclass': 0, 'Education': 1, 'Marital Status': 0, 'Occupation': 0, 'Relationship': 0,
            'Race': -2, 'Sex': -2, 'Capital Gain': 0, 'Capital Loss': 0, 'Hours per week': 0, 'Country': -2
        }
        feature_vals = {
            'Workclass': ['Never-worked', 'Self-emp-not-inc', 'Without-pay', '?', 'Private', 'Self-emp-inc',
                          'Federal-gov', 'Local-gov', 'State-gov'],
            'Education': ['High School grad', 'Dropout', 'Bachelors', 'Masters', 'Associates',
                          'Prof-School', 'Doctorate'],
            'Marital Status': ['Never-Married', 'Married', 'Separated', 'Widowed'],
            'Occupation': ['?', 'Other', 'Sales', 'Service', 'White-Collar', 'Blue-Collar', 'Professional',
                           'Admin', 'Military'],
            'Relationship': ['Not-in-family', 'Husband', 'Unmarried', 'Other-relative', 'Wife', 'Own-child'],
            'Race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
            'Sex': ['Male', 'Female'],
            'Capital Gain': ['0', '1', '2'],
            'Capital Loss': ['0', '1', '2'],
            'Country': ['United-States', 'Latin-America', 'South-America', 'China', '?', 'SE-Asia',
                        'British-Commonwealth', 'Euro_1', 'Euro_2', 'Other', 'Yugoslavia']
        }
        assert len(set(list(feature_types.keys())) - set(dataset.feature_names)) == 0, 'Inconsistent feature names'
        assert len(set(list(feature_change_restriction.keys())) - set(dataset.feature_names)) == 0, 'Inconsistent feature names'
        assert len(set(list(feature_vals.keys())) - set(dataset.feature_names)) == 0, 'Inconsistent feature names'
    elif data_name == 'compas':
        features_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        feature_names = ['Race', 'Alcohol', 'Junky', 'Supervised Release',
                         'Married', 'Felony', 'WorkRelease',
                         'Crime against Property', 'Crime against Person',
                         'Gender', 'Priors', 'YearsSchool', 'PrisonViolations',
                         'Age', 'MonthsServed', '', 'Recidivism']

        def violations_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'NO', '1': '1 to 5', '2': 'More than 5'})

        def priors_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [-1, 0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'UNKNOWN', '1': 'NO', '2': '1 to 5', '3': 'More than 5'})

        transformations = {
            0: lambda x: replace_binary_values(x, ['Black', 'White']),
            1: lambda x: replace_binary_values(x, ['No', 'Yes']),
            2: lambda x: replace_binary_values(x, ['No', 'Yes']),
            3: lambda x: replace_binary_values(x, ['No', 'Yes']),
            4: lambda x: replace_binary_values(x, ['No', 'Married']),
            5: lambda x: replace_binary_values(x, ['No', 'Yes']),
            6: lambda x: replace_binary_values(x, ['No', 'Yes']),
            7: lambda x: replace_binary_values(x, ['No', 'Yes']),
            8: lambda x: replace_binary_values(x, ['No', 'Yes']),
            9: lambda x: replace_binary_values(x, ['Female', 'Male']),
            10: lambda x: priors_fn(x),
            12: lambda x: violations_fn(x),
            13: lambda x: (x.astype(float) / 12).astype(int),
            16: lambda x: replace_binary_values(x, ['No more crimes',
                                                         'Re-arrested'])
        }
        class_map = {'No more crimes': 1, 'Re-arrested': 0}
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'recidivism/Data_1980.csv'), 16,
            feature_names=feature_names, discretize=discretize,
            features_to_use=features_to_use, balance=balance,
            feature_transformations=transformations, skip_first=True, hparams=args, cmap=class_map)
        dataset.class_names = ['Re-arrested', 'No more crimes']

        feature_types = {
            'Race': 'fixed', 'Alcohol': 'unordered', 'Junky': 'unordered', 'Supervised Release': 'unordered',
            'Married': 'unordered', 'Felony': 'unordered', 'WorkRelease': 'unordered',
            'Crime against Property': 'unordered', 'Crime against Person': 'unordered', 'Gender': 'fixed',
            'Priors': 'ordered', 'YearsSchool': 'ordered', 'PrisonViolations': 'ordered',
            'Age': 'ordered', 'MonthsServed': 'ordered'
        }
        feature_change_restriction = {
            'Race': -2, 'Alcohol': 0, 'Junky': 0, 'Supervised Release': 0,
            'Married': 0, 'Felony': 0, 'WorkRelease': 0,
            'Crime against Property': 0, 'Crime against Person': 0, 'Gender': -2,
            'Priors': 1, 'YearsSchool': 1, 'PrisonViolations': 1,
            'Age': 1, 'MonthsServed': 1
        }
        feature_vals = {
            'Race': ['Black', 'White'],
            'Alcohol': ['No', 'Yes'],
            'Junky': ['No', 'Yes'],
            'Supervised Release': ['No', 'Yes'],
            'Married': ['No', 'Married'],
            'Felony': ['No', 'Yes'],
            'WorkRelease': ['No', 'Yes'],
            'Crime against Property': ['No', 'Yes'],
            'Crime against Person': ['No', 'Yes'],
            'Gender': ['Female', 'Male'],
            'Priors': ['0', '1', '2', '3'],
            'PrisonViolations': ['0', '1', '2']
        }
        assert len(set(list(feature_types.keys())) - set(dataset.feature_names)) == 0, 'Inconsistent feature names'
        assert len(set(list(feature_change_restriction.keys())) - set(dataset.feature_names)) == 0, 'Inconsistent feature names'
        assert len(set(list(feature_vals.keys())) - set(dataset.feature_names)) == 0, 'Inconsistent feature names'
    elif data_name == 'lending':
        def filter_fn(data):
            to_remove = ['Does not meet the credit policy. Status:Charged Off',
                         'Does not meet the credit policy. Status:Fully Paid',
                         'In Grace Period', '-999', 'Current']
            for x in to_remove:
                data = data[data[:, 16] != x]
            return data

        bad_statuses = set(["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"])
        transformations = {
            16: lambda x: np.array([y in bad_statuses for y in x]).astype(int),
            19: lambda x: np.array([len(y) for y in x]).astype(int),
            6: lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
            35: lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
        }
        features_to_use = [2, 12, 13, 19, 29, 35, 51, 52, 109]
        categorical_fid = [12, 109]
        class_map = {'Good Loan': 1, 'Bad Loan': 0}
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'lendingclub/LoanStats3a_securev1.csv'),
            16, ',', features_to_use=features_to_use,
            feature_transformations=transformations, fill_na='-999',
            categorical_fid=categorical_fid, discretize=discretize,
            filter_fn=filter_fn, balance=True, hparams=args, cmap=class_map)
        dataset.class_names = ['Bad Loan', 'Good Loan']

    dataset['feature_types'] = feature_types
    dataset['feature_vals'] = feature_vals
    dataset['feature_change_restriction'] = feature_change_restriction
    return dataset

def cap_gains_fn(x):
    x = x.astype(float)
    d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                    right=True).astype('|S128')
    return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_fid=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False,
                     hparams=None, cmap=None):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_fid, consider everything < 20 as categorical"""
    data_org = copy.deepcopy(data)
    if feature_transformations is None:
        feature_transformations = {}
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except:
        import pandas
        data = pandas.read_csv(data, header=None, delimiter=delimiter, na_filter=True, dtype=str).fillna(fill_na).values
    # if 'lending' in data_org:
    #     features_not_to_use = np.ones(data.shape[1], dtype=bool)
    #     features_not_to_use[features_to_use] = False
    #     data[:, features_not_to_use] = -999
    data = data.astype(str)
    ret = Dict()

    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])

    # Encode Labels and assign variable to dict
    labels = data[:, target_idx]
    labels = np.vectorize(cmap.get)(labels)
    ret.class_target = feature_names[target_idx]

    # Subseting only features to use.
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names) if i in features_to_use])
        if categorical_fid is not None:
            categorical_fid = ([features_to_use.index(x) for x in categorical_fid])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_fid:
            categorical_fid = ([x if x < target_idx else x - 1 for x in categorical_fid])

    if categorical_fid is None:
        categorical_fid = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_fid.append(f)

    ret.ordinal_fid = [x for x in range(data.shape[1]) if x not in categorical_fid]
    ret.ordinal_fnames = [feature_names[i] for i in ret.ordinal_fid]
    ret.categorical_fid = categorical_fid
    ret.categorical_fnames = [feature_names[i] for i in categorical_fid]
    ret.feature_names = feature_names

    # Balance data
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    # Creating data splits.
    train_test_splits = ShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
    train_idx, val_idx = [x for x in train_test_splits.split(data)][0]

    test_idx = train_idx[np.random.choice(np.where(labels[train_idx] == 0)[0], hparams.max_test_samples)]
    train_idx = np.array(list(set(train_idx) - set(test_idx)))
    ret.train_idx = train_idx
    ret.val_idx = val_idx
    ret.test_idx = test_idx

    ret.data = data
    assert ret.labels[ret.test_idx].sum() == 0, 'Test samples can only be from class zero.'
    assert len(ret.test_idx) == hparams.max_test_samples
    return ret

def load_binary_data(dataname, max_test_samples, balance=True):
    if 'adult' in dataname:
        class_target = 'income'
        ordinal_fnames = ['age', 'education-num', 'hours-per-week']
        categorical_fnames = ['marital-status',  'capital-gain',  'capital-loss', 'native-country', 'occupation', 'race', 'relationship', 'sex', 'workclass']
        immutable = ['age', 'sex']
    elif 'gmc' in dataname:
        class_target = 'SeriousDlqin2yrs'
        ordinal_fnames = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                     'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                     'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
        categorical_fnames = []
        immutable = ['age']
    elif 'compas' in dataname:
        class_target = 'score'
        ordinal_fnames = ['age', 'two_year_recid', 'priors_count', 'length_of_stay']
        categorical_fnames = ['c_charge_degree', 'race', 'sex']
        immutable = ['age', 'race', 'sex']
    else:
        raise ValueError('Dataset not defined.')

    ret = Dict()
    dataframe = pd.read_csv(f"./raw/{dataname.split('_')[0]}.csv")
    if 'adult' in dataname:
        dataframe = dataframe.drop('fnlwgt', axis=1)
    labels = dataframe[class_target].values.squeeze()
    ret.labels = labels
    ret.class_target = class_target

    dataframe = dataframe.drop(class_target, axis=1)
    if dataname == 'adult_binary':
        dataframe['capital-gain'] = cap_gains_fn(dataframe['capital-gain'].values).astype(str)
        dataframe['capital-loss'] = cap_gains_fn(dataframe['capital-loss'].values).astype(str)

    data = dataframe.values

    feature_names = dataframe.columns
    ret.categorical_fid = [feature_names.get_loc(cat) for cat in categorical_fnames]
    ret.categorical_fnames = categorical_fnames
    ret.ordinal_fid = [feature_names.get_loc(cat) for cat in ordinal_fnames]
    ret.ordinal_fnames = ordinal_fnames
    ret.feature_names = feature_names
    ret.immutable = immutable

    # Balance data
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    train_test_splits = ShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
    train_idx, val_idx = [x for x in train_test_splits.split(data)][0]

    test_idx = train_idx[np.random.choice(np.where(labels[train_idx] == 0)[0], max_test_samples)]
    train_idx = np.array(list(set(train_idx) - set(test_idx)))
    ret.train_idx = train_idx
    ret.val_idx = val_idx
    ret.test_idx = test_idx

    ret.data = data
    ret.labels = labels

    assert ret.labels[ret.test_idx].sum() == 0, 'Test samples can only be from class zero.'
    assert len(ret.test_idx) == max_test_samples

    if 'adult' in dataname:
        feature_types = {
            'age': 'ordered', 'workclass': 'unordered', 'education-num': 'ordered',
             'marital-status': 'unordered', 'occupation': 'unordered', 'relationship': 'unordered',
             'race': 'fixed', 'sex': 'fixed', 'capital-gain': 'ordered',
             'capital-loss': 'ordered', 'hours-per-week': 'ordered', 'native-country': 'fixed'
        }
        feature_change_restriction = {
            'age': 1, 'workclass': 0, 'education-num': 1,
             'marital-status': 0, 'occupation': 0, 'relationship': 0,
             'race': -2, 'sex': -2, 'capital-gain': 0,
             'capital-loss': 0, 'hours-per-week': 0, 'native-country': -2
        }
        feature_vals = {
            'marital-status': ['Married', 'Non-Married'],
            'native-country': ['Non-US', 'US'],
            'occupation': ['Managerial-Specialist', 'Other'],
            'race': ['Non-White', 'White'],
            'relationship': ['Husband', 'Non-Husband'],
            'sex': ['Female', 'Male'],
            'workclass': ['Non-Private', 'Private'],
            'capital-gain': ['0', '1', '2'],
            'capital-loss': ['0', '1', '2']
        }
    elif 'gmc' in dataname:
        feature_types = {
            'RevolvingUtilizationOfUnsecuredLines': 'ordered', 'age': 'ordered',
            'NumberOfTime30-59DaysPastDueNotWorse': 'ordered', 'DebtRatio': 'ordered', 'MonthlyIncome': 'ordered',
            'NumberOfOpenCreditLinesAndLoans': 'ordered', 'NumberOfTimes90DaysLate': 'ordered',
            'NumberRealEstateLoansOrLines': 'ordered', 'NumberOfTime60-89DaysPastDueNotWorse': 'ordered',
            'NumberOfDependents': 'ordered'
        }
        feature_change_restriction = {
            'RevolvingUtilizationOfUnsecuredLines': 0, 'age': 1,
            'NumberOfTime30-59DaysPastDueNotWorse': 0, 'DebtRatio': 0, 'MonthlyIncome': 0,
            'NumberOfOpenCreditLinesAndLoans': 0, 'NumberOfTimes90DaysLate': 0,
            'NumberRealEstateLoansOrLines': 0, 'NumberOfTime60-89DaysPastDueNotWorse': 0,
            'NumberOfDependents': 0
        }
        feature_vals = {}
    elif 'compas' in dataname:
        feature_types = {
            'age': 'ordered', 'two_year_recid': 'ordered', 'c_charge_degree': 'unordered',
            'race': 'fixed', 'sex': 'fixed', 'priors_count': 'ordered', 'length_of_stay': 'ordered'
        }
        feature_change_restriction = {
             'age': 1, 'two_year_recid': 0, 'c_charge_degree': 0,
            'race': -2, 'sex': -2, 'priors_count': 1, 'length_of_stay': 1
        }
        feature_vals = {
            'c_charge_degree': ['F', 'M'],
            'race': ['African-American', 'Other'],
            'sex': ['Female', 'Male']
        }
    else:
        raise ValueError('Dataset not defined.')

    assert len(set(list(feature_types.keys())) - set(ret.feature_names)) == 0, 'Inconsistent feature names'
    assert len(set(list(feature_change_restriction.keys())) - set(ret.feature_names)) == 0, 'Inconsistent feature names'
    assert len(set(list(feature_vals.keys())) - set(ret.categorical_fnames)) == 0, 'Inconsistent categorical feature names'

    ret.feature_types = feature_types
    ret.feature_vals = feature_vals
    ret.feature_change_restriction = feature_change_restriction

    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Data IO params
    parser.add_argument("--data_name", nargs="*", default=['adult', 'compas', 'adult_binary', 'gmc_binary', 'compas_binary'],
                        choices=['adult', 'compas', 'adult_binary', 'gmc_binary', 'compas_binary', 'lending'])
    parser.add_argument('--max_test_samples', default=1000, type=int, help='maximum number of test samples')
    parser.add_argument('--seed', default=10, type=int, help='seed for np and random.')
    args = parser.parse_args()

    for data_name in args.data_name:
        np.random.seed(args.seed)
        if data_name in ['adult', 'lending', 'compas']:
            dataset = load_dataset(data_name=data_name, args=args, balance=True, dataset_folder='./raw/')
        elif 'binary' in data_name:
            dataset = load_binary_data(data_name, args.max_test_samples, balance=True)

        dump_dir = f'./final/{data_name}'
        os.makedirs(dump_dir, exist_ok=True)
        print(f'Dumping Data {data_name}')
        pickle.dump(dataset, open(f'{dump_dir}/processed_data.pkl', 'wb'))
    print('Done!')


'''
How to dump dataset. 
python process.py --data_name adult recidivism adult_binary gmc_binary compas_binary 

import pickle as pkl, numpy as np
dataname = 'adult'
dataset = pkl.load(open(f'./{dataname}/processed_data.pkl', 'rb'))

# to get unique feature values. 
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
for feat in dataset.feature_names:
    f_sort = df[feat].unique().tolist()
    f_sort.sort()
    print(feat, f_sort)

df = pd.DataFrame(ret.data, columns=ret.feature_names)
for feat in ret.feature_names:
    f_sort = df[feat].unique().tolist()
    f_sort.sort()
    print(feat, f_sort)
'''