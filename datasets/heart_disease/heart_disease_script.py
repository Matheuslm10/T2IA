__author__ = "Aryslene Santos Bitencourt [RGA: 201519060122]"
__author__ = "Felipe Alves Matos Caggi   [RGA: 201719060061]"
__author__ = "Matheus Lima Machado       [RGA: 201519060068]"

from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.classifier_evaluation import EvaluateClassifiers


if __name__ == '__main__':

    data = Normalizer('./heart-disease.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    dec_tree_combinations = [
        {'criterion': 'gini', 'max_depth': 75, 'min_samples_split': 330, 'splitter': 'random'},
        {'criterion': 'entropy', 'max_depth': 200, 'min_samples_split': 210, 'splitter': 'random'},
        {'criterion': 'entropy', 'max_depth': 225, 'min_samples_split': 70, 'splitter': 'random'},
        {'criterion': 'entropy', 'max_depth': 45, 'min_samples_split': 130, 'splitter': 'random'},
        {'criterion': 'gini', 'max_depth': 65, 'min_samples_split': 70, 'splitter': 'random'},
    ]

    knn_combinations = [
        {'algorithm': 'kd_tree', 'leaf_size': 15, 'n_neighbors': 10, 'weights': 'uniform'},
        {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 10, 'weights': 'uniform'},
        {'algorithm': 'kd_tree', 'leaf_size': 45, 'n_neighbors': 10, 'weights': 'uniform'},
        {'algorithm': 'kd_tree', 'leaf_size': 40, 'n_neighbors': 10, 'weights': 'uniform'},
        {'algorithm': 'kd_tree', 'leaf_size': 35, 'n_neighbors': 10, 'weights': 'uniform'}
    ]

    log_reg_combinations = [
        {'C': 20.255019392306675, 'max_iter': 1500, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 1.7834308769319096, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l1', 'solver': 'liblinear'},
        {'C': 15.167168884709232, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 15.167168884709232, 'max_iter': 1500, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 15.167168884709232, 'max_iter': 2000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
    ]

    mlp_combinations = [
        {'hidden_layer_sizes': 11, 'max_iter': 500, 'random_state': 9, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 11, 'max_iter': 500, 'random_state': 8, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 11, 'max_iter': 100, 'random_state': 8, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 9, 'max_iter': 500, 'random_state': 6, 'solver': 'lbfgs'},
        {'hidden_layer_sizes': 9, 'max_iter': 100, 'random_state': 8, 'solver': 'lbfgs'},
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
