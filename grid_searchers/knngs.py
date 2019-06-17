__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold


class KNNGS:
    ordered_results = None
    best_results = None

    def __init__(self, x, y, skf):
        k_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        weights_options = ['uniform', 'distance']
        algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]

        param_grid = dict(n_neighbors=k_range, weights=weights_options, algorithm=algorithm_options, leaf_size=leaf_size_range)
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        raw_results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = raw_results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        self.ordered_results = ordered_results
        self.best_results =  target_combinations
