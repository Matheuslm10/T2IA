__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

from datasets.data_normalizer import DataNormalizer as Normalizer
from sklearn.model_selection import StratifiedKFold
from grid_searchers.dec_tree_gs import DecTreeGS
from grid_searchers.knngs import KNNGS
from grid_searchers.log_reg_gs import LogRegGS
from grid_searchers.mlpgs import MLPGS
from grid_searchers.naive_bayes_gs import NaiveBayes_GS


if __name__ == '__main__':

    data = Normalizer('./car-evaluation.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    # clf_gs = KNNGS(x, y, skf)
    # clf_gs = DecTreeGS(x, y, skf)
    # clf_gs = LogRegGS(x, y, skf)
    # clf_gs = MLPGS(x, y, skf)
    clf_gs = NaiveBayes_GS(x, y, skf)

    ordered_results = clf_gs.ordered_results
    best_results = clf_gs.best_results
