import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from Combination import Combination


class MLPUtils:

    @staticmethod
    def find_best_parameters(x, y, skf):
        solver_options = ['lbfgs']
        max_iter_options = [100, 200, 400, 800, 1000, 2000]
        random_state_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        param_grid = dict(solver=solver_options, max_iter=max_iter_options, random_state=random_state_options)
        grid = GridSearchCV(MLPClassifier(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        return ordered_results, target_combinations

    @staticmethod
    def get_combinations(target_combinations):
        combinations = []

        # Go through the 5 combinations
        for row in range(1, target_combinations.shape[0] + 1):
            dictionary = target_combinations.iloc[(row - 1):row, 2:3]['params'].values[0]
            solver = dictionary.get('solver')
            max_iter = dictionary.get('max_iter')
            random_state = dictionary.get('random_state')
            classifier_algorithm = MLPClassifier(solver=solver, max_iter=max_iter, random_state=random_state)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations
