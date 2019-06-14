import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

def encontrarMelhorParametro(nome, modelo, X, y):

    skf = StratifiedKFold(n_splits=10, shuffle=False)

    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)

    grid = GridSearchCV(modelo, param_grid, cv=skf, scoring='accuracy', return_train_score=False)
    grid.fit(X, y)
    #print(pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']])
    print(nome)
    print('MELHOR ACURARICA: ',grid.best_score_)
    print('MELHOR PARÂMETRO: ',grid.best_params_)
    print('MELHOR ESTIMADOR: ',grid.best_estimator_)

def calcularMediaAcuracia(nome, modelo, X, y):


    encontrarMelhorParametro(nome, modelo, X, y)

    #acuraciaPorFold = cross_val_score(modelo, X, y, cv=skf)
    #print(nome)
    #print("Acurácias:", scores)
    #print("Acurácia (Média):", np.mean(acuraciaPorFold))
    #print()


def lerDataset():
    arquivo = open('iris.data', 'r')

    dados = []
    for linha in arquivo.readlines():
        linha = linha.replace('\n', '')
        dados.append(linha.split(','))

    del (dados[-1])

    for linha in dados:
        if (linha[-1] == 'Iris-setosa'):
            linha[-1] = '0'
        elif (linha[-1] == 'Iris-versicolor'):
            linha[-1] = '1'
        elif (linha[-1] == 'Iris-virginica'):
            linha[-1] = '2'

    dados = np.array(dados).astype(float)

    return dados


if __name__ == '__main__':
    dados = lerDataset()

    X = dados[:, 0:4]
    y = dados[:, [4]]
    y.shape = (150,)

    #calcularMediaAcuracia("REGRESSAO LOGISTICA:", LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))

    calcularMediaAcuracia("KNN:", KNeighborsClassifier(), X, y)

    #calcularMediaAcuracia("ARVORE DE DECISAO:", tree.DecisionTreeClassifier())

    #calcularMediaAcuracia("REDES NEURAIS MLP:", MLPClassifier(max_iter=1000))


