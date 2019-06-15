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


def calcularMediaAcuracia(nome, modelo):
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    acuraciaPorFold = cross_val_score(modelo, X, y, cv=skf)
    print(modelo)
    print("Acurácias:", acuraciaPorFold)
    print("Acurácia (Média):", np.mean(acuraciaPorFold))
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
        modelo = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size, n_neighbors=n_neighbors, weights=weights)



    # calcularMediaAcuracia("REGRESSAO LOGISTICA:", LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))

    # calcularMediaAcuracia("ARVORE DE DECISAO:", tree.DecisionTreeClassifier())

    # calcularMediaAcuracia("REDES NEURAIS MLP:", MLPClassifier(max_iter=1000))
