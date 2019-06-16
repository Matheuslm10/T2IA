import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import tree


class DecTreeGS:
    ordered_results = None

    def __init__(self, x, y, skf):
        criterion_options = ['gini', 'entropy']
        splitter_options = ['best', 'random']
        max_depth_range = list(range(1, 11))

        param_grid = dict(criterion=criterion_options, splitter=splitter_options, max_depth=max_depth_range)
        grid = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=skf, scoring='accuracy',
                            return_train_score=False)
        grid.fit(x, y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

        self.ordered_results = results.sort_values('mean_test_score', ascending=False)