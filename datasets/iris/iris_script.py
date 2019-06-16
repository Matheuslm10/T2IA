from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.classifier_evaluation import EvaluateClassifiers


if __name__ == '__main__':

    data = Normalizer('./iris.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    dec_tree_combinations = [
        {'criterion': 'gini', 'splitter': 'best', 'max_depth': 8},
        {'criterion': 'gini', 'splitter': 'best', 'max_depth': 8},
        {'criterion': 'gini', 'splitter': 'best', 'max_depth': 8},
        {'criterion': 'gini', 'splitter': 'best', 'max_depth': 8},
        {'criterion': 'gini', 'splitter': 'best', 'max_depth': 8},
    ]
    knn_combinations = [
        {'n_neighbors': 1, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 1},
        {'n_neighbors': 1, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 1},
        {'n_neighbors': 1, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 1},
        {'n_neighbors': 1, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 1},
        {'n_neighbors': 1, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 1},
    ]

    log_reg_combinations = [
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto'},
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto'},
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto'},
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto'},
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto'},
    ]
    mlp_combinations = [
        {'solver': 'lbfgs', 'max_iter': 400, 'random_state': 6},
        {'solver': 'lbfgs', 'max_iter': 400, 'random_state': 6},
        {'solver': 'lbfgs', 'max_iter': 400, 'random_state': 6},
        {'solver': 'lbfgs', 'max_iter': 400, 'random_state': 6},
        {'solver': 'lbfgs', 'max_iter': 400, 'random_state': 6},
    ]
    naive_bayes_combinations = [
        {'alpha': 0.8, 'fit_prior': True},
        {'alpha': 0.8, 'fit_prior': True},
        {'alpha': 0.8, 'fit_prior': True},
        {'alpha': 0.8, 'fit_prior': True},
        {'alpha': 0.8, 'fit_prior': True},
    ]

    evaluate_classifiers = EvaluateClassifiers()
    evaluate_classifiers.dec_tree_combinations = dec_tree_combinations
    evaluate_classifiers.knn_combinations = knn_combinations
    evaluate_classifiers.log_reg_combinations = log_reg_combinations
    evaluate_classifiers.mlp_combinations = mlp_combinations
    evaluate_classifiers.naive_bayes_combinations = naive_bayes_combinations

    evaluate_classifiers.evaluate_classifiers(x, y)
