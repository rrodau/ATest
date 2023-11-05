import pandas as pd
import numpy as np
import argparse
import glob
from copy import copy
import json


def detect_exp_number(path):
    """
    Instead of splitting the dataset over multiple columns, we can just detect the experiment number by finding
    the folder name. This is quite dirty and relies on a specific folder structure, which is however also hardcoded
    in run_gridsearch.py. This also means that the script can only be called from the present directory. Could
    be changed in the future if it becomes relevant.
    """
    return int(path.split('/')[2])


def json_to_rows(d_path, exp_num=-1, add_col=None):
    with open(d_path, 'r') as f:
        d = json.load(f)
        rand_d = get_rand_chance_subdict(d)
        ckpts = d['checkpoints']
        hp = d['hyperparameters']
        rows = []
        for key, values in ckpts.items():
            row = {}
            row['step'] = int(key)
            tmp = copy(values['metrics'])
            row = {**row, **tmp, **hp, **rand_d, 'dataset': d['dataset']}
            if exp_num > -1:
                row['exp_num'] = exp_num
            if add_col is not None:
                row['split'] = add_col
            rows.append(row)
        return rows


def get_rand_chance_subdict(d):
    return {k: v for k, v in d.items() if 'rand' in k}


def add_metric_column(df, metric_fun, col_name='metric'):
    metric_col = metric_fun(df)
    df[col_name] = metric_col
    return df


def simple_metric(df):
    return df['dr_ndcg'] + (1 - df['dr_rnd'])


def representation_metric_acc(df):
    lr_diff = np.abs(df['lr_s_acc'] - df['s_val_rand'])
    rf_diff = np.abs(df['rf_s_acc'] - df['s_val_rand'])
    return pd.concat([lr_diff, rf_diff], axis=1).max(axis=1)


def representation_metric_rnd(df):
    return 1 - (df.loc[:, ['lr_y_rnd', 'rf_y_rnd']]).min(axis=1)


def representation_metric_ndcg(df):
    return df.loc[:, ['lr_y_ndcg', 'rf_y_ndcg']].max(axis=1)


def representation_metric(df):
    return representation_metric_acc(df) + representation_metric_rnd(df) + representation_metric_ndcg(df)


def representation_metric_fair(df):
    return representation_metric_acc(df) + representation_metric_ndcg(df)


def all_measures_metric(df):
    return representation_metric(df) + simple_metric(df)


def acc_metric(df):
    return 1 - representation_metric_acc(df)


def all_measures_weighted_metric(df):
    ranking_metric = df["dr_ndcg"] + df.loc[:, ['lr_y_ndcg', 'rf_y_ndcg']].max(axis=1)
    fair_metric = (1 - df['dr_rnd']) + (1 - representation_metric_acc(df)) + representation_metric_rnd(df)
    return ranking_metric * 5 / 2 + fair_metric * 5 / 3

def gpa_weighted_metric(df):
    ranking_metric = df["dr_ndcg"] + df.loc[:, ['lr_y_ndcg', 'rf_y_ndcg']].max(axis=1)
    fair_metric = (1 - df['dr_gpa']) + (1 - representation_metric_acc(df)) + representation_metric_rnd(df)
    return ranking_metric * 5 / 2 + fair_metric * 5 / 3

def get_repr_csv(results_list):
    final_table = []
    old_dict = {}
    for r_idx in range(1, len(results_list)):
        cur_dict = {}
        for r_i_dx, row in enumerate(results_list[r_idx]):
            if r_idx > 1:
                cur_dict[r_i_dx] = np.concatenate([old_dict[r_i_dx], row[1:]])
            else:
                cur_dict[r_i_dx] = np.concatenate([results_list[0][r_i_dx], row[1:]])

        if r_idx > 1:
            max_key_old = max(old_dict, key=int)
            max_key_cur = max(cur_dict, key=int)
            if max_key_old > max_key_cur:
                cur_dict[max_key_old] = np.concatenate([old_dict[max_key_old], ["-", "-", "-"]])
        old_dict = cur_dict

    for key in old_dict:
        final_table.append(old_dict[key])
    if final_table != []:
        np.savetxt('newplots/repr_results.csv', final_table, fmt='%s', delimiter=",")
    return final_table


def wiki_metric(df):
    return df["dr_auc"].apply(lambda x: 1 - x if x < 0.5 else x) + (1 - df["dr_gpa"])


def get_result_df(grid_paths, type='validation'):
    glob_str = '{}/*/val_results.json'
    test_str = '{}/*/test_results.json'
    result_list_val = []
    result_list_test = []
    for path in grid_paths:
        result_list_val += glob.glob(glob_str.format(path))
        result_list_test += glob.glob(test_str.format(path))
    test_row_dict_list = []
    val_row_dict_list = []
    for val_result, test_result in zip(result_list_val, result_list_test):
        exp_num = detect_exp_number(test_result)
        test_row_dict_list.extend(json_to_rows(test_result, exp_num=exp_num, add_col='test'))
        val_row_dict_list.extend(json_to_rows(val_result, exp_num=exp_num, add_col='validation'))
    val_df = pd.DataFrame(val_row_dict_list)
    test_df = pd.DataFrame(test_row_dict_list)
    df = pd.concat((val_df, test_df))
    return df


def get_dataset_subset(df, dataset_name):
    return df[df['dataset'] == dataset_name]


def get_model_subset(df, model_name, debug_data=False):
    if debug_data:
        return df[(df['name'] == model_name) & (df['gamma'] > 0)]
    else:
        return df[(df['name'] == model_name) & (df['gamma'] > 0) & (df['step'] >= 1000)]


def get_noise_subset(df, value):
    return df[(df['noise_module'] == value) & (df['split'] == 'validation')]


def get_baseline_subset(df):
    return df[(df['gamma'] == 0) & (df['split'] == 'validation')]


def remove_nostep_subset(df):
    return df[df['step'] > 0]


def get_test_row(df, val_row):
    test_row = df.loc[(df['step'] == val_row['step']) & (df['exp_num'] == val_row['exp_num']) & (df['split'] == 'test')
                      & (df['save_dir'] == val_row['save_dir'])]
    return test_row.iloc[0]


def most_frequent(list):
    return max(set(list), key=list.count)


def get_best_all_models(df, dataset_name, metric_name, with_baseline=False, debug_data=False):
    df = get_dataset_subset(df, dataset_name)
    # needed since the debugging grid file has only 100 steps
    if remove_nostep_subset(df).empty == False:
        df = remove_nostep_subset(df)
    model_names = ['DirectRankerAdv', 'DirectRankerAdvFlip', 'DirectRankerSymFlip', 'FairListNet', 'DebiasClassifier']
    noise_df = get_noise_subset(df, 1)
    nonoise_df = get_noise_subset(df, 0)
    hp_dict = {
        "hidden_layers": [],
        "bias_layers": [],
        "dataset": [],
        "max_steps": [],
        "gamma": [],
        "noise_type": [],
        "name": []
    }
    d = {}
    for name in model_names:
        try:
            noise_model_best = \
            get_model_subset(noise_df, name, debug_data=debug_data).sort_values(by=[metric_name], ascending=False).iloc[
                0]
            noise_model_best = get_test_row(df, noise_model_best)
            if name not in ['FairListNet', 'DebiasClassifier']:
                hp_dict["name"].append(name)
                for hp in hp_dict:
                    try:
                        hp_dict[hp].append(str(noise_model_best[hp]))
                    except:
                        print("No hp {} for model {}.".format(hp, name))
        except IndexError:
            print('Warning: a model subset is empty')
            continue
        try:
            nonoise_model_best = get_model_subset(nonoise_df, name, debug_data=debug_data).sort_values(by=[metric_name],
                                                                                                       ascending=False).iloc[
                0]
            nonoise_model_best = get_test_row(df, nonoise_model_best)
            if name not in ['FairListNet', 'DebiasClassifier']:
                hp_dict["name"].append(name)
                for hp in hp_dict:
                    try:
                        hp_dict[hp].append(str(noise_model_best[hp]))
                    except:
                        print("No hp {} for model {}.".format(hp, name))
                hp_dict["noise_type"].append("no noise")
        except IndexError:
            print('Warning: a model subset is empty')
            continue
        d['noise_{}'.format(name)] = noise_model_best
        d['nonoise_{}'.format(name)] = nonoise_model_best
    if with_baseline:
        baseline = get_baseline_subset(df).sort_values(by=[metric_name], ascending=False).iloc[0]
        d['baseline'] = get_test_row(df, baseline)
        d['baseline'].loc['name'] = 'baseline'
    df = pd.DataFrame.from_dict(d)

    print(dataset_name)
    if bool([a for a in hp_dict.values() if a != []]):
        for hp in hp_dict:
            print(hp + " " + most_frequent(hp_dict[hp]))

    return df
