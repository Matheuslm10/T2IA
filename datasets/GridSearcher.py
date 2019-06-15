import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

class GridSearcher:

    def findBestParametersForKNN(X, y, skf):
        k_range = list(range(1, 3))
        weights_options = ['uniform', 'distance']
        algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size_range = list(range(1, 2))

        param_grid = dict(n_neighbors=k_range, weights=weights_options, algorithm=algorithm_options, leaf_size=leaf_size_range)
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(X, y)
        resultados = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        resultados_ordenados = resultados.sort_values('mean_test_score', ascending=False)
        combinacoesAlvo = resultados_ordenados.iloc[:5, 0:3]
        #print('MELHOR ACURARICA: ', grid.best_score_)
        #print('MELHOR PARÂMETRO: ', grid.best_params_)
        #print('MELHOR ESTIMADOR: ', grid.best_estimator_)

        return resultados_ordenados, combinacoesAlvo
