import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from Combination import Combination


class NaiveBayesUtils:

    @staticmethod
    def find_best_parameters(x, y, skf):
        alpha_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fit_prior_options = [True, False]

        param_grid = dict(alpha=alpha_options, fit_prior=fit_prior_options)
        grid = GridSearchCV(MultinomialNB(), param_grid, cv=skf, scoring='accuracy', return_train_score=False)
        grid.fit(x, y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
        ordered_results = results.sort_values('mean_test_score', ascending=False)
        target_combinations = ordered_results.iloc[:5, 0:3]

        return ordered_results, target_combinations

    @staticmethod
    def get_combinations(target_combinations):
        combinations = []

        # Go through the 5 combinations
        for row in range(1, target_combinations.shape[0] + 1):
            dictionary = target_combinations.iloc[(row - 1):row, 2:3]['params'].values[0]
            alpha = dictionary.get('alpha')
            fit_prior = dictionary.get('fit_prior')
            classifier_algorithm = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations
