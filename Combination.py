from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB


class Combination:

    classifier = None
    accuracies_list = []
    mean_accuracy = None
    std_accuracy = None
    log_loss_list = []
    mean_log_loss = None
    std_log_loss = None

    def __init__(self, classifier, accuracies_list, mean_accuracy, std_accuracy, log_loss_list, mean_log_loss,
                 std_log_loss):
        self.classifier = classifier
        self.accuracies_list = accuracies_list
        self.mean_accuracy = mean_accuracy
        self.std_accuracy = std_accuracy
        self.log_loss_list = log_loss_list
        self.mean_log_loss = mean_log_loss
        self.std_log_loss = std_log_loss

    @staticmethod
    def get_dec_tree_combinations(target_combinations):
        combinations = []

        # Go through the 5 combinations
        for dictionary in target_combinations:
            criterion = dictionary.get('criterion')
            splitter = dictionary.get('splitter')
            max_depth = dictionary.get('max_depth')
            min_samples_split = dictionary.get('min_samples_split')
            classifier_algorithm = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations

    @staticmethod
    def get_knn_combinations(target_combinations):
        combinations = []

        # Go through the 5 combinations
        for dictionary in target_combinations:
            algorithm = dictionary.get('algorithm')
            leaf_size = dictionary.get('leaf_size')
            n_neighbors = dictionary.get('n_neighbors')
            weights = dictionary.get('weights')
            classifier_algorithm = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size,
                                                        n_neighbors=n_neighbors,
                                                        weights=weights)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations

    @staticmethod
    def get_log_reg_combinations(target_combinations):
        combinations = []

        # Go through the 5 combinations
        for dictionary in target_combinations:
            C = dictionary.get('C')
            max_iter = dictionary.get('max_iter')
            multi_class = dictionary.get('multi_class')
            penalty = dictionary.get('penalty')
            solver = dictionary.get('solver')

            classifier_algorithm = LogisticRegression(C=C, max_iter=max_iter, multi_class=multi_class, penalty=penalty, solver=solver)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations

    @staticmethod
    def get_mlp_combinations(target_combinations):
        combinations = []

        # Go through the 5 combinations
        for dictionary in target_combinations:
            solver = dictionary.get('solver')
            max_iter = dictionary.get('max_iter')
            random_state = dictionary.get('random_state')
            hidden_layer_sizes = dictionary.get('hidden_layer_sizes')
            classifier_algorithm = MLPClassifier(solver=solver, max_iter=max_iter, random_state=random_state, hidden_layer_sizes=hidden_layer_sizes)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations

    @staticmethod
    def get_naive_bayes_combinations(target_combinations):
        combinations = []

        # Go through the 5 combinations
        for dictionary in target_combinations:
            alpha = dictionary.get('alpha')
            fit_prior = dictionary.get('fit_prior')
            classifier_algorithm = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            accuracy_list, log_list = [], []
            comb = Combination(classifier_algorithm, accuracy_list, None, None, log_list, None, None)
            combinations.append(comb)

        return combinations
