__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.classifier_evaluation import EvaluateClassifiers


if __name__ == '__main__':

    data = Normalizer('./wine.data').ready_data

    x = data[:, 1:]
    y = data[:, 0]
    y.shape = (len(x),)

    dec_tree_combinations = [
        {'criterion': 'gini', 'max_depth': 135, 'min_samples_split': 10, 'splitter': 'random'},
        {'criterion': 'gini', 'max_depth': 115, 'min_samples_split': 10, 'splitter': 'random'},
        {'criterion': 'gini', 'max_depth': 270, 'min_samples_split': 10, 'splitter': 'random'},
        {'criterion': 'entropy', 'max_depth': 175, 'min_samples_split': 10, 'splitter': 'random'},
        {'criterion': 'entropy', 'max_depth': 160, 'min_samples_split': 10, 'splitter': 'random'},
    ]
    knn_combinations = [
        {'algorithm': 'ball_tree', 'leaf_size': 40, 'n_neighbors': 10, 'weights': 'distance'},
        {'algorithm': 'kd_tree', 'leaf_size': 65, 'n_neighbors': 10, 'weights': 'distance'},
        {'algorithm': 'ball_tree', 'leaf_size': 45, 'n_neighbors': 15, 'weights': 'distance'},
        {'algorithm': 'kd_tree', 'leaf_size': 60, 'n_neighbors': 15, 'weights': 'distance'},
        {'algorithm': 'kd_tree', 'leaf_size': 60, 'n_neighbors': 15, 'weights': 'uniform'}
    ]


    log_reg_combinations = [
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto', 'max_iter': 1000, 'C': 325.508859983506},
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto', 'max_iter': 2000, 'C': 488.02515836544336},
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto', 'max_iter': 1500, 'C': 580.5225516094902},
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto', 'max_iter': 1000, 'C': 580.5225516094902},
        {'penalty': 'l1', 'solver': 'liblinear', 'multi_class': 'auto', 'max_iter': 2000, 'C': 547.8901179593945},
    ]
    mlp_combinations = [
        {'hidden_layer_sizes': 10, 'max_iter': 500, 'random_state': 6, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 9, 'max_iter': 100, 'random_state': 7, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 9, 'max_iter': 500, 'random_state': 7, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 10, 'max_iter': 100, 'random_state': 6, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 11, 'max_iter': 100, 'random_state': 6, 'solver': 'lbfgs'},
    ]
    naive_bayes_combinations = [
        {'alpha': 0.01, 'fit_prior': True},
        {'alpha': 0.01, 'fit_prior': False},
        {'alpha': 1.0, 'fit_prior': True},
        {'alpha': 0.9, 'fit_prior': False},
        {'alpha': 0.9, 'fit_prior': True}
    ]

    eval_clfs = EvaluateClassifiers()
    eval_clfs.dec_tree_combinations = dec_tree_combinations
    eval_clfs.knn_combinations = knn_combinations
    eval_clfs.log_reg_combinations = log_reg_combinations
    eval_clfs.mlp_combinations = mlp_combinations
    eval_clfs.naive_bayes_combinations = naive_bayes_combinations

    eval_clfs.evaluate_classifiers(x, y)
