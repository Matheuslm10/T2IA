from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.evaluate_classifier import EvaluateClassifier


if __name__ == '__main__':

    data = Normalizer('./wine.data').ready_data

    x = data[:, 1:]
    y = data[:, 0]
    y.shape = (len(x),)

    EvaluateClassifier(x, y)
