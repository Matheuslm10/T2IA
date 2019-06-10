import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets


def calcularMediaAcuracia(nome, modelo):
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    acuraciaPorFold = cross_val_score(modelo, X, y, cv=skf)
    print(nome)
    #print("Acurácias:", scores)
    print("Acurácia (Média):", np.mean(acuraciaPorFold))
    print()

if __name__ == '__main__':
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    calcularMediaAcuracia("REGRESSAO LOGISTICA:", LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))

    calcularMediaAcuracia("KNN:", KNeighborsClassifier())

    calcularMediaAcuracia("ARVORE DE DECISAO:", tree.DecisionTreeClassifier())

    calcularMediaAcuracia("REDES NEURAIS MLP:", MLPClassifier(max_iter=1000))


