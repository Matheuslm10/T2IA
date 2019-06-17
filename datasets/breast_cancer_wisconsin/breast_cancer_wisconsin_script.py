__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.classifier_evaluation import EvaluateClassifiers


if __name__ == '__main__':

    data = Normalizer('./breast-cancer-wisconsin.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    dec_tree_combinations = [
        {'criterion': 'gini', 'max_depth': 30, 'min_samples_split': 50, 'splitter': 'random'},
        {'criterion': 'gini', 'max_depth': 285, 'min_samples_split': 30, 'splitter': 'random'},
        {'criterion': 'entropy', 'max_depth': 275, 'min_samples_split': 50, 'splitter': 'random'},
        {'criterion': 'entropy', 'max_depth': 110, 'min_samples_split': 50, 'splitter': 'random'},
        {'criterion': 'entropy', 'max_depth': 270, 'min_samples_split': 50, 'splitter': 'random'},
    ]

    knn_combinations = [
        {'algorithm': 'auto', 'leaf_size': 25, 'n_neighbors': 5, 'weights': 'distance'},
        {'algorithm': 'ball_tree', 'leaf_size': 20, 'n_neighbors': 5, 'weights': 'distance'},
        {'algorithm': 'kd_tree', 'leaf_size': 20, 'n_neighbors': 5, 'weights': 'distance'},
        {'algorithm': 'kd_tree', 'leaf_size': 35, 'n_neighbors': 5, 'weights': 'distance'},
        {'algorithm': 'ball_tree', 'leaf_size': 35, 'n_neighbors': 5, 'weights': 'distance'},
    ]

    log_reg_combinations = [
        {'C': 1.0, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'solver': 'liblinear'},
        {'C': 4.500557675700499, 'max_iter': 2000, 'multi_class': 'auto', 'penalty': 'l1', 'solver': 'liblinear'},
        {'C': 30.36771118035459, 'max_iter': 2000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 30.36771118035459, 'max_iter': 1500, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 30.36771118035459, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
    ]

    mlp_combinations = [
        {'hidden_layer_sizes': 10, 'max_iter': 500, 'random_state': 9, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 10, 'max_iter': 100, 'random_state': 9, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 11, 'max_iter': 500, 'random_state': 8, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 11, 'max_iter': 100, 'random_state': 8, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 9, 'max_iter': 100, 'random_state': 9, 'solver': 'lbfgs'},
    ]

    naive_bayes_combinations = [
        {'alpha': 0.01, 'fit_prior': True},
        {'alpha': 0.3, 'fit_prior': True},
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
