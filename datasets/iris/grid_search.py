from datasets.data_normalizer import DataNormalizer as Normalizer
from sklearn.model_selection import StratifiedKFold
from grid_searchers.dectree_gs import DECTREE_GS
from grid_searchers.knn_gs import KNN_GS
from grid_searchers.logreg_gs import LOGREG_GS
from grid_searchers.mlp_gs import MLP_GS
from grid_searchers.naivebayes_gs import NAIVEBAYES_GS


if __name__ == '__main__':

    data = Normalizer('./iris.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    # clf_gs = KNN_GS(x, y, skf)
    clf_gs = DECTREE_GS(x, y, skf)
    # clf_gs = LOGREG_GS(x, y, skf)
    # clf_gs = MLP_GS(x, y, skf)
    # clf_gs = NAIVEBAYES_GS(x, y, skf)

    results = clf_gs.ordered_results
    combinations = clf_gs.target_combinations
