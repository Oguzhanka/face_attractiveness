"""
Data processing class. Reads the raw data from the folder and
generates mini-batches and labels.
"""
from skimage import transform, io
import numpy as np
import pathlib
import random
import time


class DataProcessor:
    """
    Data processing class implementation.
    """
    def __init__(self, data_type):
        """
        Initialization.
        :param data_type: Type of the split of the data. Choose train, validation or test.
        """
        if data_type == "train":
            self.data_list = list(pathlib.Path("images/training/").glob('*'))
        elif data_type == "validation":
            self.data_list = list(pathlib.Path("images/validation/").glob('*'))
        else:
            self.data_list = list(pathlib.Path("images/test/").glob('*'))

        self.data = []
        self.label = []

        self.__read_data_to_tensor()
        self.data_copy = []
        self.label_copy = []

        self.middle_data = None
        self.middle_label = None

        self.cursor = 0

    def get_data(self):
        """
        Stacks the list of data and labels read from the files.
        :return: Tensor of images.
        """
        return np.stack(self.data, axis=0), np.stack(self.label, axis=0)[:, None]

    def init_random_batches(self, batch_size):
        """
        Creates random mini-batches from the read and transformed data.
        :param batch_size: Size of the random mini-batches.
        :return: None.
        """
        self.__reset_data()
        self.cursor = 0
        index = self.__get_index(self.data)
        random.shuffle(index)
        self.middle_data = list()
        self.middle_label = list()
        for item in range(len(index)):
            self.middle_data.append(self.data[index[item]])
            self.middle_label.append(self.label[index[item]])

        self.data_copy = self.__divide_batches(self.middle_data, batch_size)
        self.label_copy = self.__divide_batches(self.middle_label, batch_size)

    def next_batches(self):
        """
        Returns the next mini-batch. If there are no mini-batches left, returns None.
        :return: (batch_x, batch_y)
        """
        self.cursor += 1
        if self.has_next():
            return self.data_copy[self.cursor - 1], np.array(self.label_copy[self.cursor - 1])[:, None]
        else:
            return None, None

    def has_next(self):
        """
        Flag to check whether there are any more batches to return.
        :return: Flag.
        """
        return self.cursor != len(self.data_copy)

    def __reset_data(self):
        """
        Stores the current data to an additional array.
        :return: None.
        """
        self.data_copy = self.data.copy()
        self.label_copy = self.label.copy()

    def __read_data_to_tensor(self):
        """
        Reads the raw data from files to a single list of tensors.
        :return: None.
        """
        start_time = time.time()
        for image_path in self.data_list:
            self.data.append(self.__decode_img(io.imread(str(image_path))))
            self.label.append(self.__get_label(image_path.stem))

        whole_data = np.stack(self.data, axis=0)
        data_mean = whole_data.mean(axis=0)
        data_std = whole_data.std(axis=0)
        norm_data = (whole_data - data_mean) / data_std
        self.data = [data for data in norm_data]
        print("Reading images takes %s\n" % (time.time() - start_time))

    @staticmethod
    def __get_index(data):
        """
        Returns the indices of the current data list.
        :param data: list of image tensors.
        :return: List of indices.
        """
        return list(range(len(data)))

    @staticmethod
    def __divide_batches(data, batch_size):
        """
        Divides and returns the batches for the specified batch size.
        :param data: List of image tensors.
        :param batch_size: Batch size for mini-batches.
        :return: Mini-batch.
        """
        data = np.stack(data, axis=0)
        return [data[i:i + batch_size] for i in range(0, int(data.shape[0]), batch_size - 1)]

    @staticmethod
    def __decode_img(img):
        """
        Decodes the image arrays to float type with fixed size of 80x80.
        :param img: Image array.
        :return: Decoded image array.
        """
        img = img.astype(float)
        return transform.resize(img, [80, 80])

    @staticmethod
    def __get_label(file_path):
        """
        Reads the label from the corresponding file path of the image.
        :param file_path: File path of the image.
        :return: label.
        """
        parts = file_path.split("_")[0]
        return parts
