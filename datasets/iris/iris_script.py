import numpy as np

from datasets.DataNormalizer import DataNormalizer as normalizer
from classifiers_utils.KNNUtils import KNNUtils
from classifiers_utils.LogisticRegressionUtils import LogisticRegressionUtils
from classifiers_utils.DecisionTreeUtils import DecisionTreeUtils
from classifiers_utils.MLPUtils import MLPUtils
from classifiers_utils.NaiveBayesUtils import NaiveBayesUtils

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


def evaluate_classifier(classifier_combinations, X, y):

    for comb in classifier_combinations:
        comb.classifier.fit(X, y)

    skf = StratifiedKFold(n_splits=10, shuffle=False)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for comb in classifier_combinations:
            prediction = comb.classifier.predict(X_test)
            prediction_prob = comb.classifier.predict_proba(X_test)

            comb.accuracys_list.append(accuracy_score(y_test, prediction))
            comb.log_loss_list.append(log_loss(y_test, prediction_prob))

    for comb in classifier_combinations:
        print()
        print(comb.classifier)
        comb.mean_accuracy = np.mean(comb.accuracys_list)
        comb.std_accuracy = np.std(comb.accuracys_list)
        print("acuracias: ", comb.accuracys_list)
        print("Acurácia (média):", comb.mean_accuracy)
        print("Desvio padrão da Acurácia:", comb.std_accuracy)

        comb.mean_log_loss = np.mean(comb.log_loss_list)
        comb.std_log_loss = np.std(comb.log_loss_list)
        print("Log loss (média):", comb.mean_log_loss)
        print("Desvio padrão do Log Loss:", comb.std_log_loss)


if __name__ == '__main__':

    dados = normalizer('./iris.data').ready_data

    X = dados[:, 0:4]
    y = dados[:, [4]]
    y.shape = (150,)

    skf = StratifiedKFold(n_splits=10, shuffle=False)

    results, target_combinations = KNNUtils.findBestParameters(X, y, skf)
    knn_combinations = KNNUtils.getCombinations(target_combinations)
    evaluate_classifier(knn_combinations, X, y)

    results2, target_combinations2 = LogisticRegressionUtils.findBestParameters(X, y, skf)
    logreg_combinations = LogisticRegressionUtils.getCombinations(target_combinations2)
    evaluate_classifier(logreg_combinations, X, y)

    results3, target_combinations3 = DecisionTreeUtils.findBestParameters(X, y, skf)
    dectree_combinations = DecisionTreeUtils.getCombinations(target_combinations3)
    evaluate_classifier(dectree_combinations, X, y)

    results4, target_combinations4 = MLPUtils.findBestParameters(X, y, skf)
    mlp_combinations = MLPUtils.getCombinations(target_combinations4)
    evaluate_classifier(mlp_combinations, X, y)

    results5, target_combinations5 = NaiveBayesUtils.findBestParameters(X, y, skf)
    naivebayes_combinations = NaiveBayesUtils.getCombinations(target_combinations5)
    evaluate_classifier(naivebayes_combinations, X, y)
