# -*- coding: utf-8 -*-


from dpcm import *
from compare import *

synthetic_setting = {
    "aggregation": {
        "path": "../dataset/experiment/synthetic/aggregation/aggregation.csv",
        "save_path": "../results/synthetic/",
        "use_cols": [0, 1],
        "num": 7,
        "dc_method": 1,
        "rho_method": 1,
        "delta_method": 1
    },
    "D31": {
        "path": "../dataset/experiment/synthetic/D31/D31.csv",
        "save_path": "../results/synthetic/",
        "use_cols": [0, 1],
        "num": 31,
        "dc_method": 1,
        "rho_method": 1,
        "delta_method": 1
    },
    "dim512": {
        "path": "../dataset/experiment/synthetic/dim512/dim512.csv",
        "save_path": "../results/synthetic/",
        "use_cols": list(range(512)),
        "num": 16,
        "dc_method": 1,
        "rho_method": 1,
        "delta_method": 1
    },
    "flame": {
        "path": "../dataset/experiment/synthetic/flame/flame.csv",
        "save_path": "../results/synthetic/",
        "use_cols": [0, 1],
        "num": 2,
        "dc_method": 1,
        "rho_method": 1,
        "delta_method": 1
    },
    "jain": {
        "path": "../dataset/experiment/synthetic/jain/jain.csv",
        "save_path": "../results/synthetic/",
        "use_cols": [0, 1],
        "num": 2,
        "dc_method": 1,
        "rho_method": 1,
        "delta_method": 1
    },
    "R15": {
        "path": "../dataset/experiment/synthetic/R15/R15.csv",
        "save_path": "../results/synthetic/",
        "use_cols": [0, 1],
        "num": 15,
        "dc_method": 1,
        "rho_method": 1,
        "delta_method": 1
    },
    "s2": {
        "path": "../dataset/experiment/synthetic/s2/s2.csv",
        "save_path": "../results/synthetic/",
        "use_cols": [0, 1],
        "num": 15,
        "dc_method": 1,
        "rho_method": 1,
        "delta_method": 1
    },
    "spiral": {
        "path": "../dataset/experiment/synthetic/spiral/spiral.csv",
        "save_path": "../results/synthetic/",
        "use_cols": [0, 1],
        "num": 3,
        "dc_method": 1,
        "rho_method": 1,
        "delta_method": 1
    }
}


def run_kmeans(path, save_path="../results/", num=0, use_cols=None, params=None):
    """

    :param path:
    :param save_path:
    :param num:
    :param use_cols:
    :param params:
    :return:
    """
    # print(path, "KMeans")
    al = ComKMeans(
        path, save_path=save_path, num=num, use_cols=use_cols, params=params
    )
    al.cluster()


def run_ap(path, save_path="../results/", use_cols=None, params=None):
    """

    :param path:
    :param save_path:
    :param use_cols:
    :param params:
    :return:
    """
    # print(path, "Ap")
    al = ComAP(
        path, save_path=save_path, use_cols=use_cols, params=params
    )
    al.cluster()


def run_ms(path, save_path="../results/", use_cols=None, params=None):
    """

    :param path:
    :param save_path:
    :param use_cols:
    :param params:
    :return:
    """
    # print(path, "Ms")
    al = ComMeanShit(
        path, save_path=save_path, use_cols=use_cols, params=params
    )
    al.cluster()


def run_sc(path, save_path="../results/", num=0, use_cols=None, params=None):
    """

    :param path:
    :param save_path:
    :param num:
    :param use_cols:
    :param params:
    :return:
    """
    # print(path, "sc")
    al = ComSC(
        path, save_path=save_path, num=num, use_cols=use_cols, params=params
    )
    al.cluster()


def run_ac(path, save_path="../results/", num=0, use_cols=None, params=None):
    """

    :param path:
    :param save_path:
    :param num:
    :param use_cols:
    :param params:
    :return:
    """
    # print(path, "Ac")
    al = ComAC(
        path, save_path=save_path, num=num, use_cols=use_cols, params=params
    )
    al.cluster()


def run_db(path, save_path="../results/", use_cols=None, params=None):
    """

    :param path:
    :param save_path:
    :param use_cols:
    :param params:
    :return:
    """
    # print(path, "Db")
    al = ComDBSCAN(
        path, save_path=save_path, use_cols=use_cols, params=params
    )
    al.cluster()


def run_op(path, save_path="../results/", num=0, use_cols=None, params=None):
    """

    :param path:
    :param save_path:
    :param num:
    :param use_cols:
    :param params:
    :return:
    """
    # print(path, "Opt")
    al = ComOPTICS(
        path, save_path=save_path, num=num, use_cols=use_cols, params=params
    )
    al.cluster()


def run_bi(path, save_path="../results/", num=0, use_cols=None, params=None):
    """

    :param path:
    :param save_path:
    :param num:
    :param use_cols:
    :param params:
    :return:
    """
    # print(path, "Birch")
    al = ComBirch(
        path, save_path=save_path, num=num, use_cols=use_cols, params=params
    )
    al.cluster()


def run_dpc(path, save_path="../results/", use_cols=None, num=0, dc_method=0, dc_percent=1,
            rho_method=1, delta_method=1, distance_method='euclidean', params=None, use_halo=False):
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
    :param params:
    :param use_halo:
    :return:
    """
    # print(path, "standard dpc")
    al = Dpc(
        path, save_path, use_cols, num, dc_method, dc_percent,
        rho_method, delta_method, distance_method, use_halo
    )
    al.cluster()


def run_dpc_d(path, save_path="../results/", use_cols=None, num=0, dc_method=0, dc_percent=1,
              rho_method=1, delta_method=1, distance_method='euclidean', params=None, use_halo=False):
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
    :param params:
    :param use_halo:
    :return:
    """
    # print(path, distance_method + "_dpc")
    al = DpcD(
        path, save_path, use_cols, num, dc_method, dc_percent,
        rho_method, delta_method, distance_method, params, use_halo
    )
    al.cluster()


def run_dpc_knn(path, save_path="../results/", use_cols=None, num=0, dc_method=0, dc_percent=1,
                rho_method=1, delta_method=1, distance_method='euclidean', params=None, use_halo=False):
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
    :param params:
    :param use_halo:
    :return:
    """
    # print(path, "Dpc knn")
    al = DpcKnn(
        path, save_path, use_cols, num, dc_method, dc_percent,
        rho_method, delta_method, distance_method, params, use_halo
    )
    al.cluster()


def run_dpc_i_rho(path, save_path="../results/", use_cols=None, num=0, dc_method=0, dc_percent=1,
                  rho_method=1, delta_method=1, distance_method='irod', params=None, use_halo=False):
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
    :param params:
    :param use_halo:
    :return:
    """
    # print(path, "Dpc_iRho")
    al = DpcIRho(
        path, save_path, use_cols, num, dc_method, dc_percent,
        rho_method, delta_method, distance_method, params, use_halo
    )
    al.cluster()


def run_snn_dpc(path, save_path="../../results/", use_cols=None, num=0, k=3):
    """

    :param path:
    :param save_path:
    :param use_cols:
    :param num:
    :param k:
    :return:
    """
    # print(path, "Snn dpc")
    al = SnnDpc(
        path, save_path, use_cols, num, k
    )
    al.cluster()


def run_i_dpc(path, save_path="../../results/", use_cols=None, num=0, dc_method=0, dc_percent=1,
              rho_method=3, delta_method=1, distance_method='euclidean', params=None, use_halo=False):
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
    :param params:
    :param use_halo:
    :return:
    """
    # print(path, "iDpc")
    al = IDpc(
        path, save_path, use_cols, num, dc_method, dc_percent,
        rho_method, delta_method, distance_method, params, use_halo
    )
    al.cluster()
