from datasets.data_normalizer import DataNormalizer as Normalizer
from sklearn.model_selection import StratifiedKFold
from grid_searchers.dec_tree_gs import DecTreeGS
from grid_searchers.knngs import KNNGS
from grid_searchers.log_reg_gs import LogRegGS
from grid_searchers.mlpgs import MLPGS
from grid_searchers.naive_bayes_gs import NaiveBayes_GS


if __name__ == '__main__':

    data = Normalizer('./breast-cancer-wisconsin.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    # clf_gs = KNN_GS(x, y, skf)
    clf_gs = DecTreeGS(x, y, skf)
    # clf_gs = LOGREG_GS(x, y, skf)
    # clf_gs = MLP_GS(x, y, skf)
    # clf_gs = NAIVEBAYES_GS(x, y, skf)

    results = clf_gs.ordered_results