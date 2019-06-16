from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.evaluate_classifier import EvaluateClassifier


if __name__ == '__main__':

    data = Normalizer('./dota-2.data').ready_data

    print('Read1')

    x = data[:, 1:]
    y = data[:, 0]
    y.shape = (len(x),)

    print('Read2')

    EvaluateClassifier(x, y)
