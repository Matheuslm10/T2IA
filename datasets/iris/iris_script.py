from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.classifier_evaluation import EvaluateClassifiers


if __name__ == '__main__':

    data = Normalizer('./iris.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    dec_tree_combinations = [
        {'criterion': 'entropy', 'splitter': 'random', 'max_depth': 260, 'min_samples_split': 10},
        {'criterion': 'gini', 'splitter': 'random', 'max_depth': 85, 'min_samples_split': 30},
        {'criterion': 'entropy', 'splitter': 'random', 'max_depth': 105, 'min_samples_split': 30},
        {'criterion': 'entropy', 'splitter': 'random', 'max_depth': 190, 'min_samples_split': 10},
        {'criterion': 'entropy', 'splitter': 'random', 'max_depth': 245, 'min_samples_split': 30},
    ]
    knn_combinations = [
        {'n_neighbors': 15, 'weights': 'distance', 'algorithm': 'ball_tree', 'leaf_size': 55},
        {'n_neighbors': 15, 'weights': 'distance', 'algorithm': 'ball_tree', 'leaf_size': 30},
        {'n_neighbors': 15, 'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 20},
        {'n_neighbors': 15, 'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 40},
        {'n_neighbors': 15, 'weights': 'distance', 'algorithm': 'brute', 'leaf_size': 5},
    ]
    log_reg_combinations = [
        {'C': 243.74441501222216, 'max_iter': 1500, 'multi_class': 'auto', 'penalty': 'l2', 'solver':'liblinear'},
        {'C': 258.2618760682677, 'max_iter': 1500, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 258.2618760682677, 'max_iter': 1000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 243.74441501222216, 'max_iter': 2000, 'multi_class': 'auto', 'penalty': 'l2', 'solver': 'liblinear'},
        {'C': 10.116379797662075, 'max_iter': 2000, 'multi_class': 'auto', 'penalty': 'l1', 'solver': 'liblinear'},
    ]
    mlp_combinations = [
        {'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 3, 'hidden_layer_sizes': 8},
        {'solver': 'lbfgs', 'max_iter': 100, 'random_state': 5, 'hidden_layer_sizes': 8},
        {'solver': 'lbfgs', 'max_iter': 1500, 'random_state': 3, 'hidden_layer_sizes': 8},
        {'solver': 'lbfgs', 'max_iter': 2000, 'random_state': 3, 'hidden_layer_sizes': 8},
        {'solver': 'lbfgs', 'max_iter': 100, 'random_state': 8, 'hidden_layer_sizes': 8},
    ]
    naive_bayes_combinations = [
        {'alpha': 0.01, 'fit_prior': True},
        {'alpha': 0.01, 'fit_prior': False},
        {'alpha': 1.0, 'fit_prior': True},
        {'alpha': 0.9, 'fit_prior': False},
        {'alpha': 0.9, 'fit_prior': True},
    ]

    eval_clfs = EvaluateClassifiers()
    eval_clfs.dec_tree_combinations = dec_tree_combinations
    eval_clfs.knn_combinations = knn_combinations
    eval_clfs.log_reg_combinations = log_reg_combinations
    eval_clfs.mlp_combinations = mlp_combinations
    eval_clfs.naive_bayes_combinations = naive_bayes_combinations

    eval_clfs.evaluate_classifiers(x, y)
