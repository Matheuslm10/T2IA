__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.classifier_evaluation import EvaluateClassifiers


if __name__ == '__main__':

    data = Normalizer('./dota-2.data').ready_data

    print('Read1')

    x = data[:, 1:]
    y = data[:, 0]
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

    eval_clfs = EvaluateClassifiers()
    eval_clfs.dec_tree_combinations = dec_tree_combinations
    eval_clfs.knn_combinations = knn_combinations
    eval_clfs.log_reg_combinations = log_reg_combinations
    eval_clfs.mlp_combinations = mlp_combinations
    eval_clfs.naive_bayes_combinations = naive_bayes_combinations

    eval_clfs.evaluate_classifiers(x, y)
