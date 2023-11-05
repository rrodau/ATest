from FairRanking.helpers import NDCG_predictor_model, rnd_model_base, acc_fair_model, auc_estimator, \
    group_pairwise_accuracy, auc_sklearn
import json


def train_model(data, model):
    """
    Function to train the model, datasets can be anything but not trec since trec means query data
    :param data: train data
    :param model: model to train
    :return: trained model
    """
    x_train, s_train, y_train = data

    model.fit(
        x=x_train,
        y=y_train,
        y_bias=s_train,
        dataset='compas',
        ckpt_period=500,
        train_external=True
    )

    return model


def evaluate_model(data, model, dataset):
    """
    Function to evaluate a model
    :param data: evaluation data
    :param model: model to evaluate
    :param dataset: data information
    :return: evaluation dict
    """
    if dataset['name'] == 'wiki':
        auc_esti = auc_sklearn
        ndcg_esti = NDCG_predictor_model
        rnd_esti = rnd_model_base
        gpa_esti = group_pairwise_accuracy
    else:
        auc_esti = auc_estimator
        ndcg_esti = NDCG_predictor_model
        rnd_esti = rnd_model_base
        gpa_esti = group_pairwise_accuracy
    x_test, s_test, y_test = data
    ndcg = ndcg_esti(model, x_test, y_test, s_test)
    rnd = rnd_esti(model, x_test, y_test, s_test)
    auc = auc_esti(model, x_test, y_test)
    gpa = gpa_esti(model, x_test, y_test, s_test)
    h_test = model.get_representations(x_test)
    hyperparameters = model.to_dict()
    d = {'dr_ndcg': ndcg, 'dr_rnd': rnd, 'dr_auc': auc, 'dr_gpa': gpa, 'checkpoints': {},
         'hyperparameters': hyperparameters, 'dataset': dataset['name'],
         'y_train_rand': dataset['y_train_rand_chance'],
         'y_test_rand': dataset['y_test_rand_chance'],
         'y_val_rand': dataset['y_val_rand_chance'],
         's_train_rand': dataset['s_train_rand_chance'],
         's_test_rand': dataset['s_test_rand_chance'],
         's_val_rand': dataset['s_val_rand_chance'],
         }
    y_ranker_names = model.y_rankers.keys()
    s_ranker_names = model.s_rankers.keys()
    for checkpoint in model.checkpoint_steps:
        checkpoint_d = {'step': checkpoint}
        metrics_d = {}
        for ranker_name in y_ranker_names:
            metrics_d['{}_ndcg'.format(ranker_name)] = ndcg_esti(model.y_rankers[ranker_name][checkpoint],
                                                                 h_test, y_test, s_test)
            metrics_d['{}_rnd'.format(ranker_name)] = rnd_esti(model.y_rankers[ranker_name][checkpoint],
                                                               h_test, y_test, s_test)
            metrics_d['{}_auc'.format(ranker_name)] = auc_esti(model.y_rankers[ranker_name][checkpoint],
                                                               h_test, y_test)
            metrics_d['{}_gpa'.format(ranker_name)] = gpa_esti(model.y_rankers[ranker_name][checkpoint],
                                                               h_test, y_test, s_test)
        for ranker_name in s_ranker_names:
            metrics_d['{}_acc'.format(ranker_name)] = acc_fair_model(model.s_rankers[ranker_name][checkpoint],
                                                                     h_test, y_test, s_test)
        hp = model.to_dict()
        hp.pop('checkpoint_steps')
        ckpt_model = model.__class__(**hp)
        ckpt_model.load_model(checkpoint)
        metrics_d['dr_ndcg'] = ndcg_esti(ckpt_model, x_test, y_test, s_test)
        metrics_d['dr_rnd'] = rnd_esti(ckpt_model, x_test, y_test, s_test)
        metrics_d['dr_auc'] = auc_esti(ckpt_model, x_test, y_test)
        metrics_d['dr_gpa'] = gpa_esti(ckpt_model, x_test, y_test, s_test)
        checkpoint_d['metrics'] = metrics_d
        d['checkpoints'][checkpoint] = checkpoint_d
    return d


def write_results(path, dictionary):
    """
    Function for writing out a dict in a json format
    :param path: path for the file
    :param dictionary: dict to write out in a json format
    :return: 0
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, indent=4)
    return 0
