import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from datasets.DataNormalizer import DataNormalizer
from sklearn.model_selection import GridSearchCV


def find_best_parameter_for_logistic_regression(model, X, y, sfk):
    random_state = list(range(1, 10))
    solver = ['lbfgs']
    max_iter = [1000]
    multi_class = ['multinomial']

    param_grid = dict(random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class)
    grid = GridSearchCV(model, param_grid, cv=skf, scoring='accuracy', return_train_score=False)
    grid.fit(X, y)
    results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    ordered_results = results.sort_values('mean_test_score')

    print('MELHOR ACURARICA: ', grid.best_score_)
    print('MELHOR PARÂMETRO: ', grid.best_params_)
    print('MELHOR ESTIMADOR: ', grid.best_estimator_)
    # target_combination = ordered_results.loc['mean_test_score']
    target_combination = ordered_results
    return ordered_results, target_combination


if __name__ == '__main__':
    data = DataNormalizer('./iris.data')

    X = data.ready_data[:, 0:4]
    y = data.ready_data[:, [4]]
    y.shape = (len(X))

    skf = StratifiedKFold(n_splits=10, shuffle=False)

    model = LogisticRegression()
    results, target_combination = find_best_parameter_for_logistic_regression(model, X, y, skf)

    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    clf.fit(X, y)

    accuracy, log = [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # predict
        prediction = clf.predict(X_test)
        prediction_prob = clf.predict_proba(X_test)

        # accuracy
        accuracy.append(accuracy_score(y_test, prediction))

        # log loss
        log.append(log_loss(y_test, prediction_prob))

    print("Accuracy (average):", np.mean(accuracy))
    print("Log loss (average):", np.mean(log))

    # calcularMediaAcuracia("REGRESSAO LOGISTICA:", LogisticRegression(solver='lbfgs', max_iter=1000,
    # multi_class='multinomial'))

    # calcularMediaAcuracia("KNN:", KNeighborsClassifier())

    # calcularMediaAcuracia("ARVORE DE DECISAO:", tree.DecisionTreeClassifier())

    # calcularMediaAcuracia("REDES NEURAIS MLP:", MLPClassifier(max_iter=1000))
