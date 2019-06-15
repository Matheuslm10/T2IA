from datasets.DataNormalizer import DataNormalizer

if __name__ == '__main__':

    # Read file and capture data
    data_file = DataNormalizer('./car.data')

    # Raw data
    raw_data = data_file.raw_data

    # Dictionary containing attributes and their values
    attribute_dictionary = data_file.attribute_dictionary

    # Data processing based on the attribute dictionary
    ready_data = data_file.ready_data
