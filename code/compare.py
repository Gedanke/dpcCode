# -*- coding: utf-8 -*-


import os
import json
import pandas
from setting import *
from sklearn.cluster import KMeans, MiniBatchKMeans, \
    AffinityPropagation, MeanShift, SpectralClustering, \
    AgglomerativeClustering, DBSCAN, OPTICS, Birch


class ComBase:
    """
    sklearn base class
    """

    def __init__(
            self, path, save_path="../../results/", num=0, use_cols=None, params=None
    ):
        """
        :param path: 
        :param save_path: 
        :param num: 
        :param use_cols: 
        :param params: 
        """
        self.path = path
        self.save_path = save_path
        self.num = num
        self.data_name = os.path.splitext(os.path.split(self.path)[-1])[0]
        self.file_name = self.data_name
        if use_cols is None:
            use_cols = [0, 1]
        self.use_cols = use_cols
        if params is None:
            params = {}
        self.params = params

        '''other param'''
        self.samples = pandas.DataFrame({})
        self.label_sign = True
        self.label_true = list()
        self.label_pred = list()
        self.cluster_result = dict()

        self.init_points_msg()

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
        self.samples = numpy.array(self.samples)

    def cluster(self):
        """
        :return:
        """

    def save_result(self, name):
        """
        :param name: 
        :return:
        """
        save_samples = dict()
        save_samples["num"] = self.label_pred
        save_samples = pandas.DataFrame(save_samples)
        save_samples.to_csv(self.get_file_path("datas", name), index=False)

        self.cluster_result.update(get_result(self.samples, self.label_true, self.label_pred, self.label_sign))

        with open(self.get_file_path("results", name), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.cluster_result, ensure_ascii=False))

    def get_file_path(self, dir_type, name):
        """
        :param dir_type: 
        :param name: 
        :return:
        path: 
        """
        path = self.save_path + dir_type + "/" + self.data_name + "/"
        if not os.path.isdir(path):
            os.mkdir(path)

        path += name + "/"
        if not os.path.isdir(path):
            os.mkdir(path)

        path += self.file_name

        if dir_type == "datas":
            path += ".csv"
        else:
            path += ".json"

        return path


class ComKMeans(ComBase):
    """
    KMeans 
    """

    def __init__(
            self, path, save_path="../../results/", num=0, use_cols=None, params=None
    ):
        """
        :param path: 
        :param save_path: 
        :param num: 
        :param use_cols: 
        :param params: 
        """
        super(ComKMeans, self).__init__(
            path, save_path=save_path, num=num, use_cols=use_cols, params=params
        )

    def cluster(self):
        """
        :return:
        """
        if "MiniBatch" in self.params.keys() and self.params["MiniBatch"]:
            algorithm = MiniBatchKMeans(n_clusters=self.num)
            self.file_name = "miniBatchKMeans"
        else:
            algorithm = KMeans(n_clusters=self.num)
            self.file_name = "kmeans"

        self.label_pred = algorithm.fit_predict(self.samples)
        self.cluster_result["center"] = algorithm.cluster_centers_.tolist()

        self.save_result("kmeans")


class ComAP(ComBase):
    """
    AffinityPropagation 
    """

    def __init__(
            self, path, save_path="../../results/", use_cols=None, params=None
    ):
        """
        :param path: 
        :param save_path:
        :param use_cols:
        :param params:
        """
        super(ComAP, self).__init__(
            path, save_path=save_path, num=0, use_cols=use_cols, params=params
        )

    def cluster(self):
        """
        :return:
        """
        self.file_name = "ap"

        if "pre" in self.params:
            algorithm = AffinityPropagation(preference=self.params["pre"])
            self.file_name += "__pre_" + str(self.params["pre"])
        else:
            algorithm = AffinityPropagation()

        self.label_pred = algorithm.fit_predict(self.samples)
        self.cluster_result["center"] = [
            int(numpy.where(center == self.samples)[0][0]) for center in algorithm.cluster_centers_
        ]

        if len((set(self.label_pred))) > 1:
            self.save_result("affinityPropagation")


class ComMeanShit(ComBase):
    """
    MeanShift 
    """

    def __init__(
            self, path, save_path="../../results/", use_cols=None, params=None
    ):
        """
        :param path: 
        :param save_path:
        :param use_cols:
        :param params: 
        """
        super(ComMeanShit, self).__init__(
            path, save_path=save_path, num=0, use_cols=use_cols, params=params
        )

    def cluster(self):
        """
        :return:
        """
        if "bandwidth" in self.params.keys():
            algorithm = MeanShift(bandwidth=self.params["bandwidth"])
            self.file_name = "ms__bd_" + str(self.params["bandwidth"])
        else:
            algorithm = MeanShift()
            self.file_name = "ms"

        self.label_pred = algorithm.fit_predict(self.samples)
        self.cluster_result["center"] = algorithm.cluster_centers_.tolist()

        if len((set(self.label_pred))) > 1:
            self.save_result("meanShift")


class ComSC(ComBase):
    """
    SpectralClustering 
    affinity ==> rbf
    """

    def __init__(
            self, path, save_path="../../results/", num=0, use_cols=None, params=None
    ):
        """
        :param path:
        :param save_path:
        :param num:
        :param use_cols:
        :param params:
        """
        super(ComSC, self).__init__(
            path, save_path=save_path, num=num, use_cols=use_cols, params=params
        )

    def cluster(self):
        """
        :return:
        """
        if "gamma" in self.params.keys():
            algorithm = SpectralClustering(n_clusters=self.num, gamma=self.params["gamma"])
            self.file_name = "sc__gamma_" + str(self.params["gamma"])
        else:
            algorithm = SpectralClustering(n_clusters=self.num)
            self.file_name = "sc"

        self.label_pred = algorithm.fit_predict(self.samples)

        self.save_result("spectralClustering")


class ComAC(ComBase):
    """
    Agglomerative Clustering
    """

    def __init__(
            self, path, save_path="../../results/", num=0, use_cols=None, params=None
    ):
        """
        :param path:
        :param save_path:
        :param num:
        :param use_cols:
        :param params:
        """
        super(ComAC, self).__init__(
            path, save_path=save_path, num=num, use_cols=use_cols, params=params
        )

    def cluster(self):
        """
        :return:
        """
        if "linkage" in self.params.keys() and "affinity" in self.params.keys():
            algorithm = AgglomerativeClustering(
                n_clusters=self.num, affinity=self.params["affinity"], linkage=self.params["linkage"]
            )
            self.file_name = "ac__af_" + str(self.params["affinity"]) + "__la_" + str(self.params["linkage"])
        elif "linkage" not in self.params.keys() and "affinity" in self.params.keys():
            algorithm = AgglomerativeClustering(
                n_clusters=self.num, affinity=self.params["affinity"]
            )
            self.file_name = "ac__af_" + str(self.params["affinity"])
        elif "linkage" in self.params.keys() and "affinity" not in self.params.keys():
            algorithm = AgglomerativeClustering(
                n_clusters=self.num, linkage=self.params["linkage"]
            )
            self.file_name = "ac__la_" + str(self.params["linkage"])
        else:
            algorithm = AgglomerativeClustering(
                n_clusters=self.num
            )
            self.file_name = "ac"

        self.label_pred = algorithm.fit_predict(self.samples)

        self.save_result("agglomerativeClustering")


class ComDBSCAN(ComBase):
    """
    DBSCAN
    """

    def __init__(
            self, path, save_path="../../results/", use_cols=None, params=None
    ):
        """
        :param path:
        :param save_path:
        :param use_cols:
        :param params:
        """
        super(ComDBSCAN, self).__init__(
            path, save_path=save_path, num=0, use_cols=use_cols, params=params
        )

    def cluster(self):
        """
        :return:
        """
        if "eps" in self.params.keys() and "min_samples" in self.params.keys():
            algorithm = DBSCAN(
                eps=self.params["eps"], min_samples=self.params["min_samples"]
            )
            self.file_name = "db__eps_" + str(self.params["eps"]) + "__ms_" + str(self.params["min_samples"])
        else:
            algorithm = DBSCAN(
            )
            self.file_name = "db"

        self.label_pred = algorithm.fit_predict(self.samples)

        if len((set(self.label_pred))) > 0:
            self.save_result("dbscan")


class ComOPTICS(ComBase):
    """
    OPTICS
    """

    def __init__(
            self, path, save_path="../../results/", num=0, use_cols=None, params=None
    ):
        """
        :param path:
        :param save_path:
        :param num:
        :param use_cols:
        :param params:
        """
        super(ComOPTICS, self).__init__(
            path, save_path=save_path, num=num, use_cols=use_cols, params=params
        )

    def cluster(self):
        """
        :return:
        """
        if "eps" in self.params.keys() and "min_samples" in self.params.keys():
            algorithm = OPTICS(
                eps=self.params["eps"], min_samples=self.params["min_samples"]
            )
            self.file_name = "op__eps_" + str(self.params["eps"]) + "__ms_" + str(self.params["min_samples"])
        else:
            algorithm = OPTICS(
            )
            self.file_name = "op"

        self.label_pred = algorithm.fit_predict(self.samples)

        if len((set(self.label_pred))) > 0:
            self.save_result("optics")


class ComBirch(ComBase):
    """
    Birch
    """

    def __init__(
            self, path, save_path="../../results/", num=0, use_cols=None, params=None
    ):
        """
        :param path:
        :param save_path:
        :param num:
        :param use_cols:
        :param params:
        """
        super(ComBirch, self).__init__(
            path, save_path=save_path, num=num, use_cols=use_cols, params=params
        )

    def cluster(self):
        """
        :return:
        """
        if "threshold" in self.params.keys():
            algorithm = Birch(n_clusters=self.num, threshold=self.params["threshold"])
            self.file_name = "bi__ts_" + str(self.params["threshold"])
        else:
            algorithm = Birch(
                n_clusters=self.num
            )
            self.file_name = "bi"

        self.label_pred = algorithm.fit_predict(self.samples)

        self.save_result("birch")
