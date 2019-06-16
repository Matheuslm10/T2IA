import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


class LOGREG_GS:
    ordered_results = None
    target_combinations = None

    def __init__(self, x, y, skf):
        penalty_options = ['l1', 'l2']
        solver_options = ['liblinear']
        multi_class_options = ['auto']

        param_grid = dict(penalty=penalty_options, solver=solver_options, multi_class=multi_class_options)
        grid = GridSearchCV(LogisticRegression(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        self.ordered_results = ordered_results
        self.target_combinations = target_combinations
