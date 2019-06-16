import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from classifiers_utils.knn_utils import KNNUtils
from classifiers_utils.logistic_regression_utils import LogisticRegressionUtils
from classifiers_utils.decision_tree_utils import DecisionTreeUtils
from classifiers_utils.mlp_utils import MLPUtils
from classifiers_utils.naive_bayes_utils import NaiveBayesUtils

from Combination import Combination


def evaluate_classifier(classifier_combinations, x, y):

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

    dec_tree_combinations = None
    knn_combinations = None
    log_reg_combinations = None
    mlp_combinations = None
    naive_bayes_combinations = None

    def evaluate_classifiers(self, x, y):

        comb = Combination

        dec_tree_combinations = comb.get_dec_tree_combinations(self.dec_tree_combinations)
        evaluate_classifier(dec_tree_combinations, x, y)

        knn_combinations = comb.get_knn_combinations(self.knn_combinations)
        evaluate_classifier(knn_combinations, x, y)

        log_reg_combinations = comb.get_log_greg_combinations(self.log_reg_combinations)
        evaluate_classifier(log_reg_combinations, x, y)

        mlp_combinations = comb.get_mlp_combinations(self.mlp_combinations)
        evaluate_classifier(mlp_combinations, x, y)

        naive_bayes_combinations = comb.get_naive_bayes_combinations(self.naive_bayes_combinations)
        evaluate_classifier(naive_bayes_combinations, x, y)
