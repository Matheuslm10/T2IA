import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold


class KNN_GS:
    ordered_results = None
    target_combinations = None

    def __init__(self, x, y, skf):
        k_range = list(range(1, 4))
        weights_options = ['uniform', 'distance']
        algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size_range = list(range(1, 4))

        param_grid = dict(n_neighbors=k_range, weights=weights_options, algorithm=algorithm_options, leaf_size=leaf_size_range)
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        self.ordered_results = ordered_results
        self.target_combinations = target_combinations
