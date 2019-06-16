import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


class NAIVEBAYES_GS:
    ordered_results = None
    target_combinations = None

    def __init__(self, x, y, skf):
        alpha_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fit_prior_options = [True, False]

        param_grid = dict(alpha=alpha_options, fit_prior=fit_prior_options)
        grid = GridSearchCV(MultinomialNB(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        self.ordered_results = ordered_results
        self.target_combinations = target_combinations
