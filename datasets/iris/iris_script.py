import numpy as np
import pandas as pd
from datasets.DataNormalizer import DataNormalizer as normalizer
from datasets.GridSearcher import GridSearcher as gridsearcher

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


def avaliar(classificador, X, y):
    print(classificador)
    classificador.fit(X, y)
    accuracy_list, log_list = [], []
    skf = StratifiedKFold(n_splits=10, shuffle=False)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # predict
        prediction = classificador.predict(X_test)
        prediction_prob = classificador.predict_proba(X_test)

        # accuracy
        accuracy_list.append(accuracy_score(y_test, prediction))

        # log loss
        log_list.append(log_loss(y_test, prediction_prob))

    print("Accuracy (average):", np.mean(accuracy_list))
    print("Log loss (average):", np.mean(log_list))
    print()


if __name__ == '__main__':

    dados = normalizer('./iris.data').ready_data

    X = dados[:, 0:4]
    y = dados[:, [4]]
    y.shape = (150,)

    modelo = KNeighborsClassifier()
    skf = StratifiedKFold(n_splits=10, shuffle=False)

    resultados, combinacoesAlvo = gridsearcher.findBestParametersForKNN(X, y, skf)

    # PERCORRE AS 5 COMBINAÇÕES
    for row in range(1,combinacoesAlvo.shape[0]+1):
        print(row)
        dicionario = combinacoesAlvo.iloc[(row-1):(row), 2:3]['params'].values[0]
        algorithm = dicionario.get('algorithm')
        leaf_size = dicionario.get('leaf_size')
        n_neighbors = dicionario.get('n_neighbors')
        weights = dicionario.get('weights')
        classificador = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size, n_neighbors=n_neighbors, weights=weights)
        avaliar(classificador, X, y)



    # calcularMediaAcuracia("REGRESSAO LOGISTICA:", LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))

    # calcularMediaAcuracia("ARVORE DE DECISAO:", tree.DecisionTreeClassifier())

    # calcularMediaAcuracia("REDES NEURAIS MLP:", MLPClassifier(max_iter=1000))
