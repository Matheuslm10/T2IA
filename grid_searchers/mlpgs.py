__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np


class MLPGS:
    ordered_results = None
    best_results = None

    def __init__(self, x, y, skf):
        solver_options = ['lbfgs']
        max_iter_options = [100, 500, 1000, 1500, 2000]
        random_state_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        hidden_layer_sizes_options = np.arange(8, 12)

        param_grid = dict(solver=solver_options, max_iter=max_iter_options, random_state=random_state_options, hidden_layer_sizes=hidden_layer_sizes_options)
        grid = GridSearchCV(MLPClassifier(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        raw_results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = raw_results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        self.ordered_results = ordered_results
        self.best_results = target_combinations