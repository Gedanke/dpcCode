# -*- coding: utf-8 -*-


from param import *
from multiprocessing.pool import Pool


def test_agg(dis):
    """
    :return:
    """
    p = Pool()
    v = synthetic_setting["aggregation"]

    '''hc'''
    p.apply_async(
        run_ac, args=(v["path"], v["save_path"], v["num"], v["use_cols"])
    )
    '''k-means'''
    p.apply_async(
        run_kmeans, args=(v["path"], v["save_path"], v["num"], v["use_cols"])
    )
    '''db'''
    p.apply_async(
        run_db, args=(
            v["path"], v["save_path"], v["use_cols"],
            {"eps": float(0.1), "min_samples": int(36)}
        )
    )
    '''op'''
    p.apply_async(
        run_op, args=(
            v["path"], v["save_path"], v["num"], v["use_cols"],
            {"eps": float(0.01), "min_samples": int(25)}
        )
    )
    '''sc'''
    p.apply_async(
        run_sc, args=(v["path"], v["save_path"], v["num"], v["use_cols"], {"gamma": float(4)})
    )
    '''dpc'''
    p.apply_async(
        run_dpc, args=(
            v["path"], v["save_path"], v["use_cols"], v["num"], v["dc_method"], float(3.5),
            v["rho_method"], v["delta_method"], "euclidean", False
        )
    )
    '''dpc-knn'''
    p.apply_async(
        run_dpc_knn, args=(
            v["path"], v["save_path"], v["use_cols"], v["num"], v["dc_method"], 3.0,
            v["rho_method"], v["delta_method"], "euclidean", False
        )
    )
    '''snn-dpc'''
    p.apply_async(
        run_snn_dpc, args=(
            v["path"], v["save_path"], v["use_cols"], v["num"], int(15)
        )
    )
    '''IKNN-DPC'''
    p.apply_async(
        run_i_dpc, args=(
            v["path"], v["save_path"], v["use_cols"], v["num"], v["dc_method"], 1,
            3, v["delta_method"], "euclidean", {"k": int(24), "mu": float(20)}, False
        )
    )

    p.close()
    p.join()


if __name__ == '__main__':
    """"""
    # test_agg("")
