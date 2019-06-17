import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes_GS:
    ordered_results = None
    best_results = None

    def __init__(self, x, y, skf):
        alpha_options = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fit_prior_options = [True, False]

        param_grid = dict(alpha=alpha_options, fit_prior=fit_prior_options)
        grid = GridSearchCV(MultinomialNB(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        raw_results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = raw_results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        self.ordered_results = ordered_results
        self.best_results = target_combinations
