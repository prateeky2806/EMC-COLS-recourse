import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from tqdm import tqdm
import argparse


def get_subroup_info(queries, data, feat, thresh=0.5, invalid=99999):
    subgroups_cats = []
    str2cat = {}

    if type(queries) == pd.DataFrame:
        queries = queries.to_dict(orient='records')

    for i, q in enumerate(queries):
        assert q == data['users'][i]['user_cost']['query'], 'queries dont match'
        if q[feat] in str2cat.keys():
            subgroups_cats.append(str2cat[q[feat]])
        else:
            try:
                new_cat = max(list(str2cat.values())) + 1
            except:
                new_cat = 0
            str2cat[q[feat]] = new_cat
            subgroups_cats.append(str2cat[q[feat]])

    subgroups_cats = np.array(subgroups_cats)
    cat2str = {v: k for k, v in str2cat.items()}

    unique_cats, freqs = np.unique(subgroups_cats, return_counts=True)
    subgroups_cost = {cat: [] for cat in unique_cats}
    subgroups_satisfied = {cat: [] for cat in unique_cats}
    subgroups_coverage = {cat: [] for cat in unique_cats}
    users_cost = []

    for ii, cat in enumerate(subgroups_cats):
        if len(data['costs'][ii]) == 0:
            user_covered = 0
            user_satisfied = 0
            users_cost.append(0)
        else:
            user_covered = 1 if data['costs'][ii].min() < invalid else 0
            user_satisfied = 1 if data['costs'][ii].min() < thresh else 0
            users_cost.append(data['costs'][ii].min())
        subgroups_satisfied[cat].append(user_satisfied)
        subgroups_coverage[cat].append(user_covered)
        if user_covered == 1:
            subgroups_cost[cat].append(data['costs'][ii].min())

    subgroups_cost = {k: sorted(v) for k, v in subgroups_cost.items()}
    num_ = max(0, min([len(v) for k, v in subgroups_cost.items()]) - 1)
    for cc in cat2str.keys():
        subgroups_cost[cc] = subgroups_cost[cc][:num_]
    #         subgroups_cost[cc] = np.random.choice(subgroups_cost[cc], num_, replace=False)

    avg_subgroup_cost = {k: np.array(subgroup_cost).mean() for k, subgroup_cost in subgroups_cost.items()}
    subgroup_satisfaction = {k: np.array(subgroup_satisfied).mean() for k, subgroup_satisfied in
                             subgroups_satisfied.items()}
    subgroup_coverage = {k: np.array(subgroup_coverage).mean() for k, subgroup_coverage in subgroups_coverage.items()}

    return avg_subgroup_cost, subgroup_satisfaction, subgroup_coverage, cat2str


def get_stats_df(feat, data_path, models, thresh=1, invalid=99999):
    df_dict = []
    for model in tqdm(models):
        print(model)

        data = pkl.load(open(f'{data_path}', 'rb'))

        num_users = data['args'].num_users
        used_queries = data['queries'][:num_users]

        model_stats = get_subroup_info(used_queries, data, feat, thresh, invalid)
        avg_subgroup_cost, subgroup_satisfaction, subgroup_coverage, cat2str = model_stats

        for cat in cat2str.keys():
            df_dict.append({'model': model, feat: cat2str[cat],
                            'ASC': avg_subgroup_cost[cat],
                            f'S-FS@{thresh}': subgroup_satisfaction[cat],
                            'S-Cov': subgroup_coverage[cat]})

    disparity_analysis = pd.DataFrame.from_records(df_dict)
    disparity_analysis = disparity_analysis.sort_values(by=['model', feat])
    # disparity_analysis.set_index(['model', feat], inplace=True)
    return disparity_analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    args = parser.parse_args()

    names_mapping = {'ls': 'LS', 'pls': 'PLS', 'random': 'R', 'dice': 'D',
                     'lime_dice': 'LD', 'face_epsilon': 'FE',
                     'face_knn': 'FK', 'ar': 'AR'}

    columns_mapping = {'ls': 'LS', 'pls': 'PLS', 'random': 'R', 'dice': 'D',
                     'lime_dice': 'LD', 'face_epsilon': 'FE',
                     'face_knn': 'FK', 'ar': 'AR'}

    sex_mapping = {'Female': 'F', 'Male': 'M'}
    race_mapping = {'White': 'W', 'Non-White': 'NW'}

    thresh = 1

    disparity_sex = get_stats_df('sex', args.data_path, [args.model], thresh=thresh).round(5)
    disparity_sex['model'] = disparity_sex['model'].replace(names_mapping)
    disparity_sex['sex'] = disparity_sex['sex'].replace(sex_mapping)
    disparity_sex.model = pd.Categorical(disparity_sex.model, categories=['D', 'FE', 'FK', 'AR', 'R', 'LS', 'PLS'])
    disparity_sex = disparity_sex.sort_values(by=['model'])
    # disparity_sex[f'NEO(S-FS@{thresh})'] = disparity_sex.groupby(['model']).apply(lambda x: (x[f'S-FS@{thresh}'] - x[f'S-FS@{thresh}'].shift(1)) / ((x[f'S-FS@{thresh}'] + x[f'S-FS@{thresh}'].shift(1))/2) ).reset_index(level=0, drop=True).astype(float).round(3).fillna('-')
    disparity_sex[f'DI(S-FS@{thresh})'] = disparity_sex.groupby(['model']).apply(lambda x: x[f'S-FS@{thresh}'] / x[f'S-FS@{thresh}'].shift(1)).reset_index(level=0, drop=True).astype(float).round(3).fillna('-')
    # disparity_sex[f'NEO(S-Cov)'] = disparity_sex.groupby(['model']).apply(lambda x: (x[f'S-Cov'] - x[f'S-Cov'].shift(1)) / ((x[f'S-Cov'] + x[f'S-Cov'].shift(1))/2) ).reset_index(level=0, drop=True).astype(float).round(3).fillna('-')
    disparity_sex[f'DI(S-Cov)'] = disparity_sex.groupby(['model']).apply(lambda x: x[f'S-Cov'] / x[f'S-Cov'].shift(1)).reset_index(level=0, drop=True).astype(float).round(3).fillna('-')
    # disparity_sex = disparity_sex.round(3)


    disparity_race = get_stats_df('race', args.data_path, [args.model], thresh=thresh).round(5)
    disparity_race['model'] = disparity_race['model'].replace(names_mapping)
    disparity_race['race'] = disparity_race['race'].replace(race_mapping)
    disparity_race.model = pd.Categorical(disparity_race.model, categories=['D', 'FE', 'FK', 'AR', 'R', 'LS', 'PLS'])
    disparity_race = disparity_race.sort_values(by=['model'])
    # disparity_race[f'NEO(S-FS@{thresh})'] = disparity_race.groupby(['model']).apply(lambda x: (x[f'S-FS@{thresh}'] - x[f'S-FS@{thresh}'].shift(1)) / ((x[f'S-FS@{thresh}'] + x[f'S-FS@{thresh}'].shift(1)) / 2)).reset_index(level=0, drop=True).astype(float).round(3).fillna('-')
    disparity_race[f'DI(S-FS@{thresh})'] = disparity_race.groupby(['model']).apply(lambda x: x[f'S-FS@{thresh}'] / x[f'S-FS@{thresh}'].shift(1)).reset_index(level=0, drop=True).astype(float).round(3).fillna('-')
    # disparity_race[f'NEO(S-Cov)'] = disparity_race.groupby(['model']).apply(lambda x: (x[f'S-Cov'] - x[f'S-Cov'].shift(1)) / ((x[f'S-Cov'] + x[f'S-Cov'].shift(1)) / 2)).reset_index(level=0, drop=True).astype(float).round(3).fillna('-')
    disparity_race[f'DI(S-Cov)'] = disparity_race.groupby(['model']).apply(lambda x: x[f'S-Cov'] / x[f'S-Cov'].shift(1)).reset_index(level=0, drop=True).astype(float).round(3).fillna('-')
    disparity_race = disparity_race

    print(disparity_sex)
    print(disparity_race)