from tqdm.auto import tqdm
import numpy as np


class ModelPerformanceExplorer(object):
    """ This class implement some of properties which
    helps to find the weakness and strength of the model

    :param predictions: predictions from the trained model
    :param test: test tf record
    :param names: path where mapped is located
    """

    def __init__(self, predictions, test, names):
        self.idx_to_label = self.load_label_name(names)
        self.x, self.y = self.unpack_records(test)
        self.y_pred = np.argmax(predictions, 1)
        self.predictions = predictions

    @staticmethod
    def unpack_records(dataset):
        """ Unpack data from tf records model

        :param dataset: test tf record
        :return: two tensors with images and labels
        """
        x, y = (list(), list())
        for _x, _y in tqdm(dataset.as_numpy_iterator()):
            x.extend(_x),  y.extend(_y)
        return np.array(x), np.array(y)

    @staticmethod
    def load_label_name(label_names, sep='\t'):
        """ load class name with a identification

        :param label_names: location of idx mapper
        :param sep:
        :return:
        """
        idx_to_label = dict()
        with open(label_names) as filename_mapper:
            for line in filename_mapper:
                name, idx = line.split(sep)
                idx_to_label[int(idx)] = name
        return idx_to_label

    def extract_cases(self, mask):
        """ This method helps to extract the wrong or
        success cases from the model prediction

        :param mask:
        :return:
        """
        _predictions = self.predictions[mask]
        valid = np.take(_predictions, self.y[mask], 1)
        dt = np.max(_predictions, 1) - np.diagonal(valid)
        return np.argsort(dt)

    def wrong_cases(self):
        """ Compute the wrong cases
        :return:
        """
        errors = self.y_pred != self.y
        return self.extract_cases(errors), errors

    def success_cases(self):
        """ Compute the success cases

        :return:
        """
        success = self.y_pred == self.y
        return self.extract_cases(success), success
