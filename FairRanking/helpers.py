from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
import itertools
import time

DATASET_LIST = ['compas', 'w3c', 'crime', 'german', 'trec', 'adult', 'synth', 'law-gender', 'law-race']
logger = logging.getLogger(sys.argv[0].strip('.py'))


def nDCG_cls(estimator, X, y, at=10, trec=False, reverse=True, k=1, m=1, esti=True):
    """
    Calculates the ndcg for a given estimator
    """
    if esti:
        prediction = np.array(estimator.predict_proba(X))
    else:
        prediction = estimator
    if len(prediction[0]) > 1:
        prediction = np.max(prediction, axis=1)
    rand = np.random.random(prediction.shape)
    sorted_list = [yi for _, _, yi in sorted(zip(prediction, rand, y), reverse=reverse)]
    yref = sorted(y, reverse=reverse)
    if trec:
        DCG = 0.
        IDCG = 0.
        max_value = max(sorted_list)
        max_idx = len(sorted_list) - 1
        for i in range(at):
            exp_dcg = sorted_list[i] + at - max_value
            exp_idcg = yref[i] + at - max_value
            if exp_dcg < 0:
                exp_dcg = 0
            if exp_idcg < 0:
                exp_idcg = 0
            DCG += (2 ** exp_dcg / (k / m) - 1) / np.log2(i + 2)
            IDCG += (2 ** exp_idcg - 1) / np.log2(i + 2)
        nDCG = DCG / IDCG
        return nDCG
    else:
        DCG = 0.
        IDCG = 0.
        for i in range(min(at, len(sorted_list))):
            DCG += (2 ** sorted_list[i] - 1) / np.log2(i + 2)
            IDCG += (2 ** yref[i] - 1) / np.log2(i + 2)
        nDCG = DCG / IDCG
        return float(nDCG)


def auc_sklearn(estimator, x, y):
    """
    Calculates the auc by using the sklearn roc_auc_score function
    """
    prediction = np.array(estimator.predict_proba(x)).tolist()
    y = y.tolist()
    if len(prediction[0]) > 1:
        prediction = np.max(prediction, axis=1)
    return roc_auc_score(y, prediction)


def auc_estimator(estimator, x, y):
    '''
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    '''
    # TODO multiclass at the moment not right! So this is only valid for the wiki dataset
    prediction = np.array(estimator.predict_proba(x))

    if len(prediction[0]) > 1:
        prediction = np.max(prediction, axis=1)
    rand = np.random.random(prediction.shape)
    real_class = [yi for _, _, yi in sorted(zip(prediction, rand, y), reverse=True)]
    yref = sorted(y, reverse=True)

    TP, FP = [0.], [0.]
    for idx, c in enumerate(real_class):
        if c == yref[idx]:
            TP.append(TP[-1] + 1)
            FP.append(FP[-1])
        else:
            TP.append(TP[-1])
            FP.append(FP[-1] + 1)
    TP.append(1. if TP[-1] == 0 else TP[-1])
    FP.append(1. if FP[-1] == 0 else FP[-1])
    TP, FP = np.array(TP), np.array(FP)
    TP = TP / TP[-1] * 100
    FP = FP / FP[-1] * 100

    # Calculate AUC
    AUC = 0.
    for i in range(len(TP) - 1):
        AUC += TP[i] * (FP[i + 1] - FP[i])
    AUC /= 10000

    return AUC


def group_pairwise_accuracy(estimator, x, y, y_bias):
    """Returns the group-dependent pairwise accuracies.

    Returns the group-dependent pairwise accuracies Acc_{G_i > G_j} for each pair
    of groups G_i \in {0, 1} and G_j \in {0, 1}.

    Args:
      prediction_diffs: NumPy array of shape (#num_pairs,) containing the
                        differences in scores for each ordered pair of examples.
      paired_groups: NumPy array of shape (#num_pairs, 2) containing the protected
                     groups for the better and worse example in each pair.

    Returns:
      A NumPy array of shape (2, 2) containing the pairwise accuracies, where the
      ij-th entry contains Acc_{G_i > G_j}.
    """
    scores = np.array(estimator.predict_proba(x))

    if len(scores[0]) > 1:
        scores = np.max(scores, axis=1)

    scores = np.reshape(scores, len(scores))
    df = pd.DataFrame()
    df = df.assign(scores=scores, labels=y, groups=y_bias[:, 0], merge_key=0)
    df = df.merge(df.copy(), on="merge_key", how="outer", suffixes=("_high", "_low"))
    df = df[df.labels_high > df.labels_low]

    paired_scores = np.stack([df.scores_high.values, df.scores_low.values], axis=1)
    paired_groups = np.stack([df.groups_high.values, df.groups_low.values], axis=1)

    # Baseline
    prediction_diffs = paired_scores[:, 0] - paired_scores[:, 1]

    accuracy_matrix = np.zeros((2, 2))
    for group_high in [0, 1]:
        for group_low in [0, 1]:
            # Predicate for pairs where the better example is from group_high
            # and the worse example is from group_low.
            predicate = ((paired_groups[:, 0] == group_high) &
                         (paired_groups[:, 1] == group_low))
            # Parwise accuracy Acc_{group_high > group_low}.
            accuracy_matrix[group_high][group_low] = (
                    np.mean(prediction_diffs[predicate] > 0) +
                    0.5 * np.mean(prediction_diffs[predicate] == 0))
    return abs(accuracy_matrix[0][1] - accuracy_matrix[1][0])


def NDCG_predictor_model(estimator, x_test, y_test, y_bias_test, at=500, queries=False, k=1, m=1, use_ranker=False):
    """
    Calculates the ndcg for the ranker output
    """
    if queries:
        # for trec at 30
        ndcg_list = []
        for x_q, y_q in zip(x_test, y_test):
            ndcg_list.append(nDCG_cls(estimator, x_q, y_q, at=at, trec=True, k=k, m=m))
        return np.mean(ndcg_list)
    else:
        if use_ranker:
            model_ndcg = nDCG_cls(estimator, x_test, y_test, at=at, trec=False, k=k, m=m)
        else:
            model_ndcg = nDCG_cls(estimator, x_test, y_test, at=at, trec=False, k=k, m=m)
        return model_ndcg


def NDCG_predictor_rf(estimator, x_test, y_test, y_bias_test, at=500, queries=False, k=1, m=1, use_ranker=False):
    """
    Calculates the ndcg for the repr. random forest
    """
    if queries:
        ndcg_list = []
        for h_q, y_q in zip(x_test, y_test):
            h_test = estimator.get_representations(h_q)
            ndcg_list.append(nDCG_cls(estimator.dr_y, h_test, y_q, at=30, trec=True, k=k, m=m))
        return np.mean(ndcg_list)
    else:
        h_test = estimator.get_representations(x_test)
        if use_ranker:
            rf_ndcg = nDCG_cls(estimator.dr_y, h_test, y_test, at=30, trec=True, k=k, m=m)
        else:
            rf_ndcg = nDCG_cls(estimator.rf_y, h_test, y_test, at, trec=False, k=k, m=m)
        return rf_ndcg


def NDCG_predictor_lr(estimator, x_test, y_test, y_bias_test, at=500, queries=False, k=1, m=1, use_ranker=False):
    """
    Calculates the ndcg for the repr. linear model
    """
    if queries:
        ndcg_list = []
        for h_q, y_q in zip(x_test, y_test):
            h_test = estimator.get_representations(h_q)
            ndcg_list.append(nDCG_cls(estimator.lNet_y, h_test, y_q, at=30, trec=True, k=k, m=m))
        return np.mean(ndcg_list)
    else:
        h_test = estimator.get_representations(x_test)
        if use_ranker:
            lr_ndcg = nDCG_cls(estimator.lNet_y, h_test, y_test, at=30, trec=True, k=k, m=m)
        else:
            lr_ndcg = nDCG_cls(estimator.lr_y, h_test, y_test, at, trec=False, k=k, m=m)
        return lr_ndcg


def acc_fair_model(estimator, x_test, y_test, y_bias_test, queries=False, use_ranker=False, dataset=None):
    """
    Calculates the acc for the ranker output on the sensible attribute
    """
    if queries:
        acc_list = []
        for h_q, s_q in zip(x_test, y_bias_test):
            s_pred_rf = estimator.predict(h_q)
            rf_acc = accuracy_score(np.argmax(s_q, axis=1), s_pred_rf)
            acc_list.append(rf_acc)
        return np.mean(acc_list)
    else:
        if dataset == "wiki":
            s = y_bias_test
        else:
            s = y_bias_test[:, 0]
        s_pred_rf = estimator.predict(x_test)
        rf_acc = accuracy_score(s, s_pred_rf)
        return rf_acc


def rnd_model_base(estimator, x_test, y_test, y_bias_test, at=500, queries=False, use_ranker=False):
    """
    Calculates the rnd for the ranker forest output on the sensible attribute
    """
    if queries:
        rnd_list = []
        s_test = y_bias_test
        for x_q, s_q, y_q in zip(x_test, s_test, y_test):
            rnd_list.append(rND_estimator(estimator, x_q, s_q))
        return np.mean(rnd_list)
    else:
        s_test = y_bias_test
        rnd = rND_estimator(estimator, x_test, s_test)
        return rnd


def rND_estimator(estimator, x, s, step=10, start=10,
                  protected_group_idx=1, non_protected_group_idx=0,
                  y=None, esti=True):
    '''
    Computes the normalized Discounted Difference, which is a measure of how different are
    the proportion of members in the protected group at the top-i cutoff and in the overall
    population. Lower is better. 0 is the best possible value. Only binary protected groups
    are supported.
    '''
    if esti:
        prediction = estimator.predict_proba(x)
    else:
        prediction = estimator

    if len(np.array(prediction).shape) > 1:
        prediction = np.amax(prediction, axis=1)
    rnd = rND(prediction, s, step=step, start=start, protected_group_idx=protected_group_idx,
              non_protected_group_idx=non_protected_group_idx)

    return rnd


def rND(prediction, s, step=10, start=10, protected_group_idx=1, non_protected_group_idx=0):
    '''
    Computes the normalized Discounted Difference, which is a measure of how different are
    the proportion of members in the protected group at the top-i cutoff and in the overall
    population. Lower is better. 0 is the best possible value. Only binary protected groups
    are supported.
    '''
    s = np.asarray(s)
    if len(s.shape) > 1:
        s = s[:, 0]
    # we don't want to have uniqual size
    if len(prediction) != len(s):
        raise AssertionError(
            'len of prediction ' + str(len(prediction)) + ' and s ' + str(len(s)) + ' are uniqual'
        )
    unique, counts = np.unique(s, return_counts=True)
    count_dict_all = dict(zip(unique, counts))
    try:
        len(unique) == 2
    except AssertionError:
        logger.error('array s contains more than 2 classes.')

    keys = [protected_group_idx, non_protected_group_idx]
    for key in keys:
        if key not in count_dict_all:
            count_dict_all[key] = 0

    sorted_idx = np.argsort(np.array(prediction))[::-1]
    sorted_s = np.array(s[sorted_idx])

    # a fake sorted list of s which gives the worst possible result, used for regularization purposes
    # it is maximally discriminative, having all non-protected individuals first and then the others.
    fake_horrible_s = np.hstack(([non_protected_group_idx for i in range(count_dict_all[non_protected_group_idx])],
                                 [protected_group_idx for i in range(count_dict_all[protected_group_idx])]))

    fake_horrible_s_2 = np.hstack(([protected_group_idx for i in range(count_dict_all[protected_group_idx])],
                                   [non_protected_group_idx for i in range(count_dict_all[non_protected_group_idx])]))

    rnd = 0
    max_rnd = 0
    max_rnd_2 = 0

    for i in range(start, len(s), step):
        unique, counts = np.unique(sorted_s[:i], return_counts=True)
        count_dict_top_i = dict(zip(unique, counts))

        unique, counts = np.unique(fake_horrible_s[:i], return_counts=True)
        count_dict_reg = dict(zip(unique, counts))

        unique_2, counts_2 = np.unique(fake_horrible_s_2[:i], return_counts=True)
        count_dict_reg_2 = dict(zip(unique_2, counts_2))

        keys = [protected_group_idx, non_protected_group_idx]
        for key in keys:
            if key not in count_dict_reg:
                count_dict_reg[key] = 0
            if key not in count_dict_top_i:
                count_dict_top_i[key] = 0
        rnd += 1 / np.log2(i) * np.abs(
            count_dict_top_i[protected_group_idx] / i - count_dict_all[protected_group_idx] / len(s))
        max_rnd += 1 / np.log2(i) * np.abs(
            count_dict_reg[protected_group_idx] / i - count_dict_all[protected_group_idx] / len(s))
        max_rnd_2 += 1 / np.log2(i) * np.abs(
            count_dict_reg_2[protected_group_idx] / i - count_dict_all[protected_group_idx] / len(s))

    if max_rnd_2 > max_rnd:
        max_rnd = max_rnd_2

    return rnd / max_rnd


def transform_pairwise(X, y, s=None, subsample=0.):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    s_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        if s is not None:
            s_new.append((s[i, 0] - s[j, 0]) ** 2)
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    X_new = np.asarray(X_new)
    y_new = np.asarray(y_new).ravel()
    s_new = np.asarray(s_new)
    idx = [idx for idx in range(len(X_new))]
    if subsample > 0.0:
        idx = np.random.randint(X.shape[0], size=int(X.shape[0] * subsample))
        X_new = X_new[idx]
        y_new = y_new[idx]
        if s is not None:
            s_new = s_new[idx]

    if s is not None:
        return np.asarray(X_new), np.asarray(y_new).ravel(), idx, np.asarray(s_new)
    else:
        return np.asarray(X_new), np.asarray(y_new).ravel(), idx
