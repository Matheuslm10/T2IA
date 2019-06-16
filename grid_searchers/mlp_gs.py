import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


class MLP_GS:
    ordered_results = None
    target_combinations = None

    def __init__(self, x, y, skf):
        solver_options = ['lbfgs']
        max_iter_options = [100, 200, 400, 800, 1000, 2000]
        random_state_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        param_grid = dict(solver=solver_options, max_iter=max_iter_options, random_state=random_state_options)
        grid = GridSearchCV(MLPClassifier(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        self.ordered_results = ordered_results
        self.target_combinations = target_combinations
