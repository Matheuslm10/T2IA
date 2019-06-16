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
