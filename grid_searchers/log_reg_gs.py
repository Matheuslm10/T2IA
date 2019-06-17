__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

class LogRegGS:
    ordered_results = None
    best_results = None

    def __init__(self, x, y, skf):
        penalty_options = ['l1', 'l2']
        C_options = np.logspace(0, 5, 200)
        solver_options = ['liblinear']
        multi_class_options = ['auto']
        max_iter_options = [1000, 1500, 2000]

        param_grid = dict(penalty=penalty_options, C=C_options, solver=solver_options, multi_class=multi_class_options, max_iter=max_iter_options)
        grid = GridSearchCV(LogisticRegression(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        raw_results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = raw_results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        self.ordered_results = ordered_results
        self.best_results = target_combinations
