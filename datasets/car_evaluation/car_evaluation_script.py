__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.classifier_evaluation import EvaluateClassifiers


if __name__ == '__main__':

    data = Normalizer('./car-evaluation.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    dec_tree_combinations = [
        {'criterion': 'gini', 'splitter': 'random', 'max_depth': 65, 'min_samples_split': 10},
        {'criterion': 'entropy', 'splitter': 'random', 'max_depth': 85, 'min_samples_split': 10},
        {'criterion': 'gini', 'splitter': 'random', 'max_depth': 220, 'min_samples_split': 10},
        {'criterion': 'entropy', 'splitter': 'random', 'max_depth': 50, 'min_samples_split': 10},
        {'criterion': 'gini', 'splitter': 'random', 'max_depth': 295, 'min_samples_split': 10},
    ]
    knn_combinations = [
        {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'brute', 'leaf_size': 45},
        {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'brute', 'leaf_size': 25},
        {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'brute', 'leaf_size': 35},
        {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'brute', 'leaf_size': 10},
        {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'brute', 'leaf_size': 40},
    ]
    log_reg_combinations = [
        {'C': 1.0, 'max_iter': 1500, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 1.0, 'max_iter': 2000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 1.059560179277616, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 1.059560179277616, 'max_iter': 1500, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 1.059560179277616, 'max_iter': 2000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
    ]
    mlp_combinations = [
        {'solver': 'lbfgs', 'max_iter': 500, 'random_state': 7, 'hidden_layer_sizes': 10},
        {'solver': 'lbfgs', 'max_iter': 500, 'random_state': 9, 'hidden_layer_sizes': 11},
        {'solver': 'lbfgs', 'max_iter': 100, 'random_state': 9, 'hidden_layer_sizes': 11},
        {'solver': 'lbfgs', 'max_iter': 500, 'random_state': 7, 'hidden_layer_sizes': 11},
        {'solver': 'lbfgs', 'max_iter': 500, 'random_state': 6, 'hidden_layer_sizes': 9},
    ]
    naive_bayes_combinations = [
        {'alpha': 0.5, 'fit_prior': True},
        {'alpha': 0.4, 'fit_prior': True},
        {'alpha': 1.0, 'fit_prior': True},
        {'alpha': 0.9, 'fit_prior': True},
        {'alpha': 0.8, 'fit_prior': True},
    ]

    eval_clfs = EvaluateClassifiers()
    eval_clfs.dec_tree_combinations = dec_tree_combinations
    eval_clfs.knn_combinations = knn_combinations
    eval_clfs.log_reg_combinations = log_reg_combinations
    eval_clfs.mlp_combinations = mlp_combinations
    eval_clfs.naive_bayes_combinations = naive_bayes_combinations

    eval_clfs.evaluate_classifiers(x, y)
