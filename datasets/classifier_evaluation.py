import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from classifiers_utils.knn_utils import KNNUtils
from classifiers_utils.logistic_regression_utils import LogisticRegressionUtils
from classifiers_utils.decision_tree_utils import DecisionTreeUtils
from classifiers_utils.mlp_utils import MLPUtils
from classifiers_utils.naive_bayes_utils import NaiveBayesUtils


def evaluate_classifier(classifier_combinations, x, y, skf):

    skf = StratifiedKFold(n_splits=10, shuffle=False)

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for comb in classifier_combinations:
            comb.classifier.fit(x_train, y_train)

        for comb in classifier_combinations:
            prediction = comb.classifier.predict(x_test)
            prediction_prob = comb.classifier.predict_proba(x_test)

            comb.accuracies_list.append(accuracy_score(y_test, prediction))
            test = log_loss(y_test, prediction_prob)
            comb.log_loss_list.append(test)

    for comb in classifier_combinations:
        print()
        print(comb.classifier)
        comb.mean_accuracy = np.mean(comb.accuracies_list)
        comb.std_accuracy = np.std(comb.accuracies_list)
        print("Accuracies: ", comb.accuracies_list)
        print("Accuracy (average):", comb.mean_accuracy)
        print("Standard deviation of precision:", comb.std_accuracy)

        comb.mean_log_loss = np.mean(comb.log_loss_list)
        comb.std_log_loss = np.std(comb.log_loss_list)
        print("Log loss (average):", comb.mean_log_loss)
        print("Standard deviation of Log Loss:", comb.std_log_loss)


class EvaluateClassifiers:

    def __init__(self, x, y):

        skf = StratifiedKFold(n_splits=10, shuffle=False)

        results, target_combinations1 = KNNUtils.find_best_parameters(x, y, skf)
        knn_combinations = KNNUtils.get_combinations(target_combinations1)
        evaluate_classifier(knn_combinations, x, y, skf)

        results2, target_combinations2 = LogisticRegressionUtils.find_best_parameters(x, y, skf)
        log_reg_combinations = LogisticRegressionUtils.get_combinations(target_combinations2)
        evaluate_classifier(log_reg_combinations, x, y, skf)

        results3, target_combinations3 = DecisionTreeUtils.find_best_parameters(x, y, skf)
        dec_tree_combinations = DecisionTreeUtils.get_combinations(target_combinations3)
        evaluate_classifier(dec_tree_combinations, x, y, skf)

        results4, target_combinations4 = MLPUtils.find_best_parameters(x, y, skf)
        mlp_combinations = MLPUtils.get_combinations(target_combinations4)
        evaluate_classifier(mlp_combinations, x, y, skf)

        results5, target_combinations5 = NaiveBayesUtils.find_best_parameters(x, y, skf)
        naive_bayes_combinations = NaiveBayesUtils.get_combinations(target_combinations5)
        evaluate_classifier(naive_bayes_combinations, x, y, skf)
