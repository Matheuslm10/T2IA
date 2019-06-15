import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from Combination import Combination

class DecisionTreeUtils:

    def findBestParameters(X, y, skf):
        criterion_options = ['gini', 'entropy']
        splitter_options = ['best', 'random']
        max_depth_range = list(range(1, 11))

        param_grid = dict(criterion=criterion_options, splitter=splitter_options, max_depth=max_depth_range)
        grid = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
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
            criterion = dictionary.get('criterion')
            splitter = dictionary.get('splitter')
            max_depth = dictionary.get('max_depth')
            classifier_algorithm = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations