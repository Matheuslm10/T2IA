import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    iris = load_iris()

    X = iris.data
    y = iris.target

    modelo = KNeighborsClassifier()
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    acuraciaPorFold = cross_val_score(modelo, X, y, cv=skf)
    print("Acurácia (Média):", np.mean(acuraciaPorFold))
    print()