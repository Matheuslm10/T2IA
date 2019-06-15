import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

class Utilities:

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
        # print('MELHOR ACURARICA: ', grid.best_score_)
        # print('MELHOR PARÂMETRO: ', grid.best_params_)
        # print('MELHOR ESTIMADOR: ', grid.best_estimator_)

        return ordered_results, target_combinations

    def passParameters(target_combinations):
        knn_combinations = []

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
            knn_combinations.append(comb)

        return knn_combinations

class Combination:

    classifier = None
    accuracys_list = []
    mean_accuracy = None
    std_accuracy = None
    log_loss_list = []
    mean_log_loss = None
    std_log_loss = None

    def __init__(self, classifier, accuracys_list, mean_accuracy, std_accuracy, log_loss_list, mean_log_loss, std_log_loss):
        self.classifier = classifier
        self.accuracys_list = accuracys_list
        self.mean_accuracy = mean_accuracy
        self.std_accuracy = std_accuracy
        self.log_loss_list = log_loss_list
        self.mean_log_loss = mean_log_loss
        self.std_log_loss = std_log_loss