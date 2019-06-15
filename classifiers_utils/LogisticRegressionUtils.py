import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from Combination import Combination

class LogisticRegressionUtils:

    def findBestParameters(X, y, skf):
        penalty_options = [ 'l1', 'l2']
        solver_options = ['liblinear']
        multi_class_options = ['auto']

        param_grid = dict(penalty=penalty_options, solver=solver_options, multi_class=multi_class_options)
        grid = GridSearchCV(LogisticRegression(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(X, y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        return ordered_results, target_combinations

    def getCombinations(target_combinations):
        combinations = []

        # PERCORRE AS 5 COMBINAÇÕES
        for row in range(1, target_combinations.shape[0] + 1):
            dictionary = target_combinations.iloc[(row - 1):(row), 2:3]['params'].values[0]
            penalty = dictionary.get('penalty')
            solver = dictionary.get('solver')
            multi_class = dictionary.get('multi_class')
            classifier_algorithm = LogisticRegression(penalty=penalty, solver=solver, multi_class=multi_class)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations