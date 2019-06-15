import numpy as np


def read_file(path):
    raw_data = open(path, 'r')

    array = []
    for line in raw_data.readlines():
        line = line.replace('\n', '')
        array.append((line.split(',')))

    return np.array(array)


class DataFile:

    raw_data = None
    attribute_dictionary = None
    ready_data = None

    def __init__(self, path):
        self.raw_data = read_file(path)
        self.attribute_dictionary = self.create_dictionary()
        self.ready_data = self.process_data()

    def create_dictionary(self):
        """
            Creates a dictionary from the raw data containing attributes and their values
        :return dict: attribute dictionary
        """
        attribute_dictionary = dict()
        for index in range(0, len(self.raw_data[0])):
            col = np.array(self.raw_data[:, index])
            attribute_dictionary[index] = np.unique(col)

        return attribute_dictionary

    def process_data(self):
        """
            # Converts the data values ​​based on the attribute dictionary
        :return numpy.array: ready-to-use data
        """
        ready_data = []
        for x_index in range(0, len(self.raw_data)):
            row = []
            for y_index in range(0, len(self.raw_data[0])):
                i, = np.where(self.attribute_dictionary[y_index] == self.raw_data[x_index, y_index])
                row.append(i[0])
            ready_data.append(row)

        return np.array(ready_data)
