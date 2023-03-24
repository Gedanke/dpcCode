# -*- coding: utf-8 -*-


import numpy
from munkres import Munkres
from sklearn.metrics.cluster import *

'''
cluster_result = {
    "center": [],
    "davies_bouldin": 0.0,
    "calinski_harabasz": 0.0,
    "silhouette_coefficient": 0.0,
    "cluster_acc": 0.0,
    "rand_index": 0.0,
    "adjusted_rand_index": 0.0,
    "mutual_info": 0.0,
    "normalized_mutual_info": 0.0,
    "adjusted_mutual_info": 0.0,
    "homogeneity": 0.0,
    "completeness": 0.0,
    "v_measure": 0.0,
    "h_c_v_m": 0.0,
    "fowlkes_mallows_index": 0.0
}
'''


def cluster_acc(label_true, label_pred):
    """
    :param label_true:
    :param label_pred:
    :return:
    acc:
    """
    label_true = numpy.array(label_true)
    label_pred = numpy.array(label_pred)

    label_true_label = numpy.unique(label_true)
    label_true_num = len(label_true_label)
    label_pred_label = numpy.unique(label_pred)
    label_pred_num = len(label_pred_label)
    num = numpy.maximum(label_true_num, label_pred_num)
    matrix = numpy.zeros((num, num))

    for i in range(label_true_num):
        ind_cla_true = label_true == label_true_label[i]
        ind_cla_true = ind_cla_true.astype(float)
        for j in range(label_pred_num):
            ind_cla_pred = label_pred == label_pred_label[j]
            ind_cla_pred = ind_cla_pred.astype(float)
            matrix[i, j] = numpy.sum(ind_cla_true * ind_cla_pred)

    m = Munkres()
    index = m.compute(-matrix.T)
    index = numpy.array(index)

    c = index[:, 1]
    label_pred_new = numpy.zeros(label_pred.shape)
    for i in range(label_pred_num):
        label_pred_new[label_pred == label_pred_label[i]] = label_true_label[c[i]]

    right = numpy.sum(label_true[:] == label_pred_new[:])
    acc = right.astype(float) / (label_true.shape[0])

    return acc


def get_result(samples, label_true, label_pred, label_sign=True):
    """
    :param samples:
    :param label_true:
    :param label_pred:
    :param label_sign:
    :return:
    cluster_result:
    """
    cluster_result = dict()

    cluster_result["davies_bouldin"] = davies_bouldin_score(samples, label_pred)
    cluster_result["calinski_harabasz"] = calinski_harabasz_score(samples, label_pred)
    cluster_result["silhouette_coefficient"] = silhouette_score(samples, label_pred)

    if label_sign:
        if len(set(label_true)) == len(set(label_pred)):
            cluster_result["cluster_acc"] = cluster_acc(label_true, label_pred)
        else:
            cluster_result["cluster_acc"] = -1

        cluster_result["rand_index"] = rand_score(label_true, label_pred)
        cluster_result["adjusted_rand_index"] = adjusted_rand_score(label_true, label_pred)
        cluster_result["mutual_info"] = mutual_info_score(label_true, label_pred)
        cluster_result["normalized_mutual_info"] = normalized_mutual_info_score(label_true, label_pred)
        cluster_result["adjusted_mutual_info"] = adjusted_mutual_info_score(label_true, label_pred)
        cluster_result["homogeneity"] = homogeneity_score(label_true, label_pred)
        cluster_result["completeness"] = completeness_score(label_true, label_pred)
        cluster_result["v_measure"] = v_measure_score(label_true, label_pred)
        cluster_result["h_c_v_m"] = homogeneity_completeness_v_measure(label_true, label_pred)
        cluster_result["fowlkes_mallows_index"] = fowlkes_mallows_score(label_true, label_pred)

    return cluster_result
