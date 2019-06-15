import numpy as np
import pandas as pd
from datasets.DataNormalizer import DataNormalizer as normalizer
from datasets.GridSearcher import GridSearcher as gridsearcher
from datasets.Combination import Combination as combination
from classifiers_utils.KNNUtils import Utilities as knn_utilities
from classifiers_utils.KNNUtils import Combination as combination

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score
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

    results, target_combinations = gridsearcher.findBestParametersForKNN(X, y, skf)

    knn_combinations = []

    # PERCORRE AS 5 COMBINAÇÕES
    for row in range(1, target_combinations.shape[0] + 1):
        dictionary = target_combinations.iloc[(row - 1):(row), 2:3]['params'].values[0]
        algorithm = dictionary.get('algorithm')
        leaf_size = dictionary.get('leaf_size')
        n_neighbors = dictionary.get('n_neighbors')
        weights = dictionary.get('weights')
        classifier_algorithm = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size, n_neighbors=n_neighbors, weights=weights)
        accuracy_list, log_list = [], []
        comb = combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
        knn_combinations.append(comb)

    evaluate_classifier(knn_combinations, X, y)

    # calcularMediaAcuracia("REGRESSAO LOGISTICA:", LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))

    # calcularMediaAcuracia("ARVORE DE DECISAO:", tree.DecisionTreeClassifier())

    # calcularMediaAcuracia("REDES NEURAIS MLP:", MLPClassifier(max_iter=1000))
