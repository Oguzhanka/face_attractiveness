from skimage import transform, io
import numpy as np
import pathlib
import random
import time


class DataProcessor:

    def __init__(self, data_type):
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
        return self.data, self.label

    def init_random_batches(self, batch_size):
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
        self.cursor += 1
        if self.has_next():
            return self.data_copy[self.cursor - 1], np.array(self.label_copy[self.cursor - 1])[:, None]
        else:
            return None, None

    def has_next(self):
        return self.cursor != len(self.data_copy)

    def __reset_data(self):
        self.data_copy = self.data.copy()
        self.label_copy = self.label.copy()

    def __read_data_to_tensor(self):
        start_time = time.time()
        for image_path in self.data_list:
            self.data.append(self.__decode_img(io.imread(str(image_path))))
            self.label.append(self.__get_label(image_path.stem))

        print("Reading images takes %s\n" % (time.time() - start_time))

    @staticmethod
    def __get_index(data):
        return list(range(len(data)))

    @staticmethod
    def __divide_batches(data, batch_size):
        data = np.stack(data, axis=0)
        return [data[i:i + batch_size] for i in range(0, int(data.shape[0]), batch_size - 1)]

    @staticmethod
    def __decode_img(img):
        img = img.astype(float)
        return transform.resize(img, [80, 80])

    @staticmethod
    def __get_label(file_path):
        parts = file_path.split("_")[0]
        return parts
