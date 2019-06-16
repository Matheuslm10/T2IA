from datasets.data_normalizer import DataNormalizer as Normalizer
from datasets.classifier_evaluation import EvaluateClassifiers


if __name__ == '__main__':

    data = Normalizer('./heart-disease.data').ready_data

    x = data[:, 0:len(data[0])-1]
    y = data[:, len(data[0])-1]
    y.shape = (len(x),)

    EvaluateClassifiers(x, y)
