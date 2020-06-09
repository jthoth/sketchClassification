from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import central_crop, flip_left_right,  resize
from tensorflow import one_hot, reshape, cast
from random import random, uniform
from tensorflow import io, train
from os.path import join
from tqdm.auto import tqdm


class DatasetBuilder(object):
    """  Is an auto contained tf-records functions which
    allow us to create it given a single path

    :param path: place where the image are
    :param shape: define width and height of images
    :param labels: # of classes

    """

    def __init__(self, path, shape=(256, 256), labels=250):
        self.path, self.shape = path, shape
        self.labels = labels

    def record_writer(self, label, name, writer):
        """ Write an specific example on a writer tensor
        record

        :param label:  image label
        :param name:  image filename
        :param writer: tensor record object
        :return:
        """
        image = load_img(name, target_size=self.shape)
        feature = dict(image=_bytes_feature(
            img_to_array(image, dtype='uint8').tostring()
        ), label=_int64_feature(label))
        features = train.Features(feature=feature)
        samples = train.Example(features=features)
        writer.write(samples.SerializeToString())

    def reader(self, file_reader):
        """ Build a iterator from file object reader line by
        line

        :param file_reader: reader data
        :return:
        """
        for line in file_reader:
            snap = line.split('\t')
            yield (join(self.path, snap[0]), int(snap[-1]))

    def build_train(self, filename='train.txt'):
        """ Save on tensor records writer  from train
        data-sets

        :param filename: output file data
        :return:
        """
        output = '{}.records'.format(filename.split('.')[0])
        writer = io.TFRecordWriter(join(self.path, output))
        with open(join(self.path, filename)) as file:
            for (name, label) in tqdm(self.reader(file)):
                self.record_writer(label, name, writer)
        writer.close()

    def build_test(self, filename='test.txt'):
        """ Save on tensor records writer  from test
        data-sets

        :param filename: output file data
        :return:
        """
        writer = io.TFRecordWriter(join(self.path, 'test.records'))
        with open(join(self.path, filename)) as file:
            for (name, label) in tqdm(self.reader(file)):
                self.record_writer(label, name, writer)
        writer.close()

    @staticmethod
    def get_template():
        return dict(image=io.FixedLenFeature([], 'string'),
                    label=io.FixedLenFeature([], 'int64'))

    def decode(self, serialized_example):
        """ Load final output to neural network training procedure

        :param serialized_example: compressed example recorded
        on TFRecordWriter

        :return: image, label as one hot encoder
        """

        features = io.parse_single_example(serialized_example,
                                           self.get_template())
        image = io.decode_raw(features['image'], 'uint8')
        image = reshape(image, self.shape + (3,))
        image = cast(image, dtype='float32') / 255
        return image, one_hot(features['label'], self.labels)

    def augmentation(self, image, label, th=.90):
        """  Basic Data augmentation  used in this pipeline
        in this step only use a basic transformation

        :param image: tensor sample
        :param label: one hot label
        :param th: random value which triggered a augmentation

        :return: 0-1 tensor and one-hot label

        """

        if random() > th:
            size = uniform(th, 1.0)
            image = central_crop(image, central_fraction=size)
            image = resize(image, self.shape, antialias=True)
        return image, label

    def __call__(self, train_file='train.txt', test_file='test.txt'):
        """ Create the tensor flow records.

        :param train_file: filename train
        :param test_file: filename test
        :return:
        """
        self.build_train(filename=train_file)
        self.build_test(filename=test_file)


def _int64_feature(value):
    return train.Feature(int64_list=train.Int64List(value=[value]))


def _bytes_feature(value):
    return train.Feature(bytes_list=train.BytesList(value=[value]))


def _float_feature(value):
    return train.Feature(float_list=train.FloatList(value=[value]))
