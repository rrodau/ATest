import json
import os
import copy
import argparse
from multiprocessing import Pool
from itertools import product
from run_experiment import train_model, evaluate_model, write_results
from FairRanking.models.DirectRankerAdv import directRankerAdv
from FairRanking.models.DirectRankerAdvFlip import directRankerAdvFlip
from FairRanking.models.DirectRankerSymFlip import directRankerSymFlip
from FairRanking.models.DebiasClassifier import DebiasClassifier
from FairRanking.models.FairListNet import FairListNet
from FairRanking.datasets.compas import Compas
from FairRanking.datasets.law import Law
from FairRanking.datasets.adult import Adult
from FairRanking.datasets.wiki import Wiki
from constants import TIMESTR, GRID_ROOT
from functools import partial
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer


def load_params(d, model=None):
    """
    Reads in the grid.json and prepares the HP for the gridsearch
    :param d: path the the json file with the grid search parameters
    :param model: name of the model if None it is taken from the json
    :return: list of experiments, dataset name, prepr. function and list models
    """
    with open(d, 'r') as f:
        d = json.load(f)
    datasets = d["dataset"]
    d.pop('dataset')
    preprocess_fn = d["preprocess_fn"]
    d.pop('preprocess_fn')

    if model == None:
        models = d["model"]
        d.pop('model')
    else:
        d["model"] = [model]
        models = model
    if model not in ["DirectRankerAdvFlip", "DirectRankerAdv", "DebiasClassifier"]:
        d["bias_layers"] = [[0, 0]]
    if model not in ["DirectRankerAdvFlip", "DirectRankerSymFlip"]:
        d["fair_loss"] = [0]
    keys, values = zip(*d.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    experiments = [(str(idx), dic) for idx, dic in enumerate(experiments)]
    return experiments, datasets, preprocess_fn, models


def prepare_hyperparameters(exp_params, save_dir):
    """
    Takes the current HP and removes the model key while adding an output directory
    :param exp_params: current HP for the run_experiment function
    :param save_dir: output directory of the experiments
    :return: dict of HP
    """
    hp = copy.deepcopy(exp_params)
    hp.pop('model')
    hp['save_dir'] = save_dir
    return hp


def prepare_data(dataset_name, preprocess_str, test_size=0.2, val_size=0.2):
    """
    Creates data handler for each dataset and return the data in arrays
    :param dataset_name: name of the dataset
    :param preprocess_str: name of the preprocessing function
    :param test_size: size of the test set
    :param val_size: size of the validation set
    :return: data array with train, test and validation data and some dataset information
    """
    dataset_str = dataset_name.lower()
    if 'compas' in dataset_str:
        dataset = Compas()
    elif 'law-race' in dataset_str:
        dataset = Law('race')
    elif 'law-gender' in dataset_str:
        dataset = Law('gender')
    elif 'adult' in dataset_str:
        dataset = Adult()
    elif 'wiki' in dataset_str:
        dataset = Wiki()
    else:
        raise ValueError('Dataset identifier {} not recognized.'.format(dataset_str))
    if 'minmax' in preprocess_str:
        preprocess_fn = MinMaxScaler()
    elif 'standard' in preprocess_str:
        preprocess_fn = StandardScaler()
    elif 'quantile' in preprocess_str:
        preprocess_fn = QuantileTransformer()
    else:
        raise ValueError('Preprocessing identifier {} not recognized.'.format(preprocess_str))
    data = (dataset.get_data(preprocess_fn=preprocess_fn, test_size=test_size, val_size=val_size))
    data_info = dataset.get_dataset_info()
    return data, data_info


def run_experiment(number, parameters, out_dir, traindata, valdata, testdata, out_dir_offset, dataset):
    """
    Runs on experiment by creating the model, fitting it and evaluating it
    :param number: number of the current experiment
    :param parameters: HP for the current experiment
    :param out_dir: path of the output directory
    :param traindata: data for fitting the model
    :param valdata: data for validating the model
    :param testdata: data for testing the model
    :param out_dir_offset: offset for output directory
    :param dataset: dataset information
    :return: 0
    """
    # prepare output dirs
    target_dir = '{}/{}'.format(out_dir, number + out_dir_offset)
    os.makedirs(target_dir)
    # prepare model
    parameters = parameters[number][1]
    hyperparams = prepare_hyperparameters(parameters, target_dir)
    model_type_str = parameters['model'].lower()
    if 'adv' in model_type_str and 'flip' in model_type_str:
        model = directRankerAdvFlip(
            num_features=dataset['get_num_features'],
            num_fair_classes=dataset['get_num_fair_classes'],
            dataset=dataset['name'],
            **hyperparams
        )
    elif 'adv' in model_type_str:
        hyperparams.pop('fair_loss')
        parameters.pop('fair_loss')
        model = directRankerAdv(
            num_features=dataset['get_num_features'],
            num_fair_classes=dataset['get_num_fair_classes'],
            dataset=dataset['name'],
            **hyperparams
        )
    elif 'symflip' in model_type_str:
        hyperparams.pop('bias_layers')
        parameters.pop('bias_layers')
        model = directRankerSymFlip(
            num_features=dataset['get_num_features'],
            num_fair_classes=dataset['get_num_fair_classes'],
            dataset=dataset['name'],
            **hyperparams
        )
    elif 'list' in model_type_str:
        hyperparams.pop('bias_layers')
        parameters.pop('bias_layers')
        hyperparams.pop('fair_loss')
        parameters.pop('fair_loss')
        model = FairListNet(
            num_features=dataset['get_num_features'],
            num_fair_classes=dataset['get_num_fair_classes'],
            dataset=dataset['name'],
            **hyperparams
        )
    elif 'debias' in model_type_str:
        hyperparams.pop('fair_loss')
        parameters.pop('fair_loss')
        model = DebiasClassifier(
            num_features=dataset['get_num_features'],
            num_fair_classes=dataset['get_num_fair_classes'],
            dataset=dataset['name'],
            num_relevance_classes=dataset['num_relevance_classes'],
            **hyperparams
        )
    else:
        raise ValueError('Parameter "model" {} not recognized.'.format(model_type_str))

    # train the model on the train data
    model = train_model(traindata, model)
    # evaluate on eval set
    val_result_dict = evaluate_model(valdata, model, dataset)
    # evaluate on test set
    test_result_dict = evaluate_model(testdata, model, dataset)
    write_results(target_dir + '/test_results.json', test_result_dict)
    write_results(target_dir + '/val_results.json', val_result_dict)
    model.close_session()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a gridsearch experiment')
    parser.add_argument('-j', '--grid_path', action='store', type=str,
                        help='Path to the .json file containing the gridsearch hyperparameters.')
    parser.add_argument('-n', '--num_jobs', action='store', default=4, type=int,
                        help='How many parallel jobs to run')
    parser.add_argument('-o', '--out_path', action='store', default='{}/{}'.format(GRID_ROOT, TIMESTR), type=str,
                        help='Out dir for results')
    parser.add_argument('-np', '--noparallel', action='store_true', help='Dont use parallelism. -n will be ignored',
                        default=False)

    args = parser.parse_args()
    _, dataset, preprocess_fn, models = load_params(args.grid_path)

    counter = 0
    run_parallel_cpu = not args.noparallel
    for dset in dataset:
        if dset != 'wiki':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            run_parallel_cpu = not args.noparallel
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            del os.environ["CUDA_VISIBLE_DEVICES"]
            run_parallel_cpu = False
        for pre in preprocess_fn:
            for model in models:
                exp_list, _, _, _ = load_params(args.grid_path, model)
                data, data_info = prepare_data(dset, pre)
                run_experiment_f = partial(
                    run_experiment,
                    parameters=exp_list,
                    out_dir=args.out_path,
                    out_dir_offset=counter,
                    traindata=data[0],
                    valdata=data[1],
                    testdata=data[2],
                    dataset=data_info
                )
                if run_parallel_cpu:
                    with Pool(args.num_jobs) as p:
                        p.map(run_experiment_f, [i for i, _ in enumerate(exp_list)])
                else:
                    for i, exp in enumerate(exp_list):
                        run_experiment_f(i)
                counter = counter + len(exp_list)
