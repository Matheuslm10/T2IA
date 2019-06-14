import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


def calcularMediaAcuracia(nome, modelo):
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    acuraciaPorFold = cross_val_score(modelo, X, y, cv=skf)
    print(nome)
    #print("Acurácias:", scores)
    print("Acurácia (Média):", np.mean(acuraciaPorFold))
    print()


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

    calcularMediaAcuracia("REGRESSAO LOGISTICA:", LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))

    calcularMediaAcuracia("KNN:", KNeighborsClassifier())

    calcularMediaAcuracia("ARVORE DE DECISAO:", tree.DecisionTreeClassifier())

    calcularMediaAcuracia("REDES NEURAIS MLP:", MLPClassifier(max_iter=1000))


