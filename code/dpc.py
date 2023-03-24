# -*- coding: utf-8 -*-


import os
import math
import json
import pandas
import scipy.cluster.hierarchy as sch
from setting import *


class Dpc:
    """
    DPC base class
    """

    def __init__(
            self, path, save_path="../../results/", use_cols=None, num=0, dc_method=0,
            dc_percent=1, rho_method=1, delta_method=1, distance_method='euclidean', use_halo=False
    ):
        """
        :param path:
        :param save_path:
        :param use_cols:
        :param num:
        :param dc_method:
        :param dc_percent:
        :param rho_method:
        :param delta_method:
        :param distance_method:
        :param use_halo:
        """
        '''parameters'''
        self.path = path
        self.save_path = save_path
        self.data_name = os.path.splitext(os.path.split(self.path)[-1])[0]
        if use_cols is None:
            use_cols = [0, 1]
        self.use_cols = use_cols
        self.num = num
        self.dc_method = dc_method
        self.dc_percent = dc_percent
        self.rho_method = rho_method
        self.delta_method = delta_method
        self.distance_method = distance_method
        self.use_halo = use_halo

        '''other parameters'''
        self.center = None
        self.border_b = list()
        self.samples = pandas.DataFrame({})
        self.samples_num = 0
        self.features_num = len(use_cols)
        self.label_sign = True
        self.label_true = list()
        self.label_pred = list()
        self.dis_matrix = pandas.DataFrame({})
        self.cluster_result = dict()

        self.file_name = "dcm_" + str(self.dc_method) + "__dcp_" + str(self.dc_percent) + \
                         "__rho_" + str(self.rho_method) + "__dem_" + str(self.delta_method) + \
                         "__ush_" + str(int(self.use_halo))
        self.algorithm_name = "dpc"

    def cluster(self):
        """
        :return:
        """
        self.init_points_msg()
        dis_array, min_dis, max_dis = self.load_points_msg()
        dc = self.get_dc(dis_array, min_dis, max_dis)
        rho = self.get_rho(dc)

        delta = self.get_delta(rho)
        gamma = self.get_center(rho, delta)
        self.cluster_result["center"] = self.center.tolist()
        cluster_result = self.assign_samples(rho, self.center)
        if self.use_halo:
            cluster_result, halo = self.get_halo(rho, cluster_result, dc)
            self.cluster_result["halo"] = halo

        self.gain_label_pred(cluster_result)
        self.save_result()

        # return rho

    def init_points_msg(self):
        """
        :return:
        """
        self.samples = pandas.read_csv(self.path)

        col = list(self.samples.columns)
        if len(col) == len(self.use_cols):
            self.label_sign = False
        else:
            self.label_sign = True
            self.label_true = self.samples[col[-1]].tolist()

        self.samples = self.samples.iloc[:, self.use_cols]
        self.samples_num = len(self.samples)
        self.dis_matrix = pandas.DataFrame(numpy.zeros((self.samples_num, self.samples_num)))

    def load_points_msg(self):
        """
        :return:
        """
        '''euclidean'''
        return self.distance_standard("euclidean")

    def distance_standard(self, distance_method):
        """
        :param distance_method:
        :return:
        dis_array:
        min_dis:
        max_dis:
        """
        dis_array = sch.distance.pdist(self.samples, distance_method)

        num = 0
        for i in range(self.samples_num):
            for j in range(i + 1, self.samples_num):
                self.dis_matrix.at[i, j] = dis_array[num]
                self.dis_matrix.at[j, i] = self.dis_matrix.at[i, j]
                num += 1

        min_dis = self.dis_matrix.min().min()
        max_dis = self.dis_matrix.max().max()

        return dis_array, min_dis, max_dis

    def get_dc(self, dis_array, min_dis, max_dis):
        """
        :param dis_array:
        :param min_dis:
        :param max_dis:
        :return:
        dc:
        """
        lower = self.dc_percent / 100
        upper = (self.dc_percent + 1) / 100

        if self.dc_method == 0:
            while 1:
                dc = (min_dis + max_dis) / 2
                neighbors_percent = len(dis_array[dis_array < dc]) / (((self.samples_num - 1) ** 2) / 2)

                if lower <= neighbors_percent <= upper:
                    return dc
                elif neighbors_percent > upper:
                    max_dis = dc
                elif neighbors_percent < lower:
                    min_dis = dc
        elif self.dc_method == 1:
            dis_array_ = dis_array.copy()
            dis_array_.sort()
            dc = dis_array_[int(float(self.dc_percent / 100.0) * self.samples_num * (self.samples_num - 1) / 2)]

            return dc

    def get_rho(self, dc):
        """
        :param dc:
        :return:
        rho:
        """
        rho = numpy.zeros(self.samples_num)

        for i in range(self.samples_num):
            if self.rho_method == 0:
                rho[i] = len(self.dis_matrix.loc[i, :][self.dis_matrix.loc[i, :] < dc]) - 1
            elif self.rho_method == 1:
                for j in range(self.samples_num):
                    if i != j:
                        rho[i] += math.exp(-(self.dis_matrix.at[i, j] / dc) ** 2)
            elif self.rho_method == 2:
                n = int(self.samples_num * 0.05)
                rho[i] = math.exp(-(self.dis_matrix.loc[i].sort_values().values[:n].sum() / (n - 1)))

        return rho

    def get_delta(self, rho):
        """
        :param rho:
        :return:
        delta:
        """
        delta = numpy.zeros(self.samples_num)

        if self.delta_method == 0:
            for i in range(self.samples_num):
                rho_i = rho[i]
                j_list = numpy.where(rho > rho_i)[0]
                if len(j_list) == 0:
                    delta[i] = self.dis_matrix.loc[i, :].max()
                else:
                    min_dis_idx = self.dis_matrix.loc[i, j_list].idxmin()
                    delta[i] = self.dis_matrix.at[i, min_dis_idx]
        elif self.delta_method == 1:
            rho_order_idx = rho.argsort()[-1::-1]
            for i in range(1, self.samples_num):
                rho_idx = rho_order_idx[i]
                j_list = rho_order_idx[:i]
                min_dis_idx = self.dis_matrix.loc[rho_idx, j_list].idxmin()
                delta[rho_idx] = self.dis_matrix.at[rho_idx, min_dis_idx]
            delta[rho_order_idx[0]] = delta.max()

        return delta

    def get_center(self, rho, delta):
        """
        :param rho:
        :param delta:
        :return:
        center:
        gamma: rho * delta
        """
        gamma = rho * delta

        gamma = pandas.DataFrame(gamma, columns=["gamma"]).sort_values("gamma", ascending=False)
        if self.num > 0:
            self.center = numpy.array(gamma.index)[:self.num]
        else:
            '''other way'''
            # center = gamma[gamma.gamma > threshold].loc[:, "gamma"].index

        return gamma

    def assign_samples(self, rho, center):
        """
        :param rho: 局
        :param center:
        :return:
        cluster: dict(center: str, points: list())
        """
        return self.assign(rho, center)

    def assign(self, rho, center):
        """
        :param rho:
        :param center:
        :return:
        cluster: dict(center: str, points: list())
        """
        cluster_result = dict()
        for c in center:
            cluster_result[c] = list()

        link = dict()
        order_rho_idx = rho.argsort()[-1::-1]
        for i, v in enumerate(order_rho_idx):
            if v in center:
                link[v] = v
                continue
            rho_idx = order_rho_idx[:i]
            link[v] = self.dis_matrix.loc[v, rho_idx].sort_values().index.tolist()[0]

        for k, v in link.items():
            c = v
            while c not in center:
                c = link[c]
            cluster_result[c].append(k)

        """
        for i in range(self.samples_num):
            c = self.dis_matrix.loc[i, center].idxmin()
            cluster_result[c].append(i)
        """

        return cluster_result

    def get_halo(self, rho, cluster_result, dc):
        """
        :param rho:
        :param cluster_result:
        :param dc:
        :return:
        cluster_result:
        halo:
        """
        all_points = set(list(range(self.samples_num)))

        for c, points in cluster_result.items():
            others_points = list(set(all_points) - set(points))
            border = list()
            for point in points:
                if self.dis_matrix.loc[point, others_points].min() < dc:
                    border.append(point)
            if len(border) == 0:
                continue

            # rbo_b = rho[border].max()
            point_b = border[rho[border].argmax()]
            self.border_b.append(point_b)
            rho_b = rho[point_b]
            filter_points = numpy.where(rho >= rho_b)[0]
            points = list(set(filter_points) & set(points))
            cluster_result[c] = points

        cluster_points = set()
        for c, points in cluster_result.items():
            cluster_points = cluster_points | set(points)

        halo = list(set(all_points) - cluster_points)

        return cluster_result, halo

    def gain_label_pred(self, cluster_result):
        """
        :param cluster_result:
        :return:
        """
        self.label_pred = [-1 for _ in range(self.samples_num)]

        if self.label_sign:
            label_true_list = list(set(self.label_true))
        else:
            label_true_list = list(range(1, self.num + 1))

        idx = 0
        for c, points in cluster_result.items():
            for point in points:
                self.label_pred[point] = label_true_list[idx]
            idx += 1

    def save_result(self):
        """
        :return:
        """
        save_samples = dict()
        save_samples["num"] = self.label_pred
        save_samples = pandas.DataFrame(save_samples)
        save_samples.to_csv(self.get_file_path("datas"), index=False)

        self.cluster_result.update(get_result(self.samples, self.label_true, self.label_pred, self.label_sign))
        with open(self.get_file_path("results"), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.cluster_result, ensure_ascii=False))

    def get_file_path(self, dir_type):
        """
        :param dir_type:
        :return:
        path:
        """
        path = self.save_path + dir_type + "/" + self.data_name + "/"
        if not os.path.isdir(path):
            os.mkdir(path)

        path += self.algorithm_name + "/"
        if not os.path.isdir(path):
            os.mkdir(path)

        path += self.file_name

        if dir_type == "datas":
            path += ".csv"
        else:
            path += ".json"

        return path
