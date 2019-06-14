import numpy as np
import pandas as pd
from datasets.DataNormalizer import DataNormalizer as normalizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


def findBestParametersForKNN(modelo, X, y, sfk):
    k_range = list(range(1, 3))
    weights_options = ['uniform', 'distance']
    algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size_range = list(range(1, 3))

    param_grid = dict(n_neighbors=k_range, weights=weights_options, algorithm=algorithm_options, leaf_size=leaf_size_range)
    grid = GridSearchCV(modelo, param_grid, cv=skf, scoring='accuracy', return_train_score=False)
    grid.fit(X, y)
    resultados = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    resultados_ordenados = resultados.sort_values('mean_test_score')


    print('MELHOR ACURARICA: ', grid.best_score_)
    print('MELHOR PARÂMETRO: ', grid.best_params_)
    print('MELHOR ESTIMADOR: ', grid.best_estimator_)
    combinacoesAlvo = resultados_ordenados.loc['mean_test_score']
    return resultados_ordenados, combinacoesAlvo


def calcularMediaAcuracia(nome, modelo):
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    acuraciaPorFold = cross_val_score(modelo, X, y, cv=skf)
    print(nome)
    # print("Acurácias:", scores)
    print("Acurácia (Média):", np.mean(acuraciaPorFold))
    print()


if __name__ == '__main__':
    dados = normalizer('./iris.data').ready_data


    X = dados[:, 0:4]
    y = dados[:, [4]]
    y.shape = (150,)


    skf = StratifiedKFold(n_splits=10, shuffle=False)

    modelo = KNeighborsClassifier()

    resultados, combinacoesAlvo = findBestParametersForKNN(modelo, X, y, skf)

    # calcularMediaAcuracia("REGRESSAO LOGISTICA:", LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))

    # calcularMediaAcuracia("ARVORE DE DECISAO:", tree.DecisionTreeClassifier())

    # calcularMediaAcuracia("REDES NEURAIS MLP:", MLPClassifier(max_iter=1000))
