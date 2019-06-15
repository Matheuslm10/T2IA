import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from Combination import Combination

class KNNUtils:

    def findBestParameters(X, y, skf):
        k_range = list(range(1, 41))
        weights_options = ['uniform', 'distance']
        algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size_range = list(range(1, 21))

        param_grid = dict(n_neighbors=k_range, weights=weights_options, algorithm=algorithm_options,
                          leaf_size=leaf_size_range)
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
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
            algorithm = dictionary.get('algorithm')
            leaf_size = dictionary.get('leaf_size')
            n_neighbors = dictionary.get('n_neighbors')
            weights = dictionary.get('weights')
            classifier_algorithm = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size, n_neighbors=n_neighbors,
                                                        weights=weights)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations