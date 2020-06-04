from tensorflow.io import TFRecordWriter, parse_single_example, FixedLenFeature, decode_raw
from tensorflow.train import Feature, BytesList, Int64List, FloatList, Example, Features
from tensorflow.image import central_crop, flip_left_right, rot90, resize
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import one_hot, reshape, cast
from random import random
from os.path import join
from tqdm import tqdm


class DatasetBuilder(object):
    """  Is an auto contained tf-records functions which
    allow us to create it given a single path

    :param path: place where the image are
    :param shape: define width and height of image

    """

    def __init__(self, path, shape=(256, 256), classes=250):
        self.path, self.shape = path, shape
        self.classes = classes

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
        sample = Example(features=Features(feature=feature))
        writer.write(sample.SerializeToString())

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
        writer = TFRecordWriter(join(self.path, 'train.records'))
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
        writer = TFRecordWriter(join(self.path, 'test.records'))
        with open(join(self.path, filename)) as file:
            for (name, label) in tqdm(self.reader(file)):
                self.record_writer(label, name, writer)
        writer.close()

    @staticmethod
    def get_template():
        return dict(image=FixedLenFeature([], 'string'),
                    label=FixedLenFeature([], 'int64'))

    def decoder(self, serialized_example):
        """ Load final output to neural network training procedure

        :param serialized_example: compressed example recorded
        on TFRecordWriter

        :return: image, label as one hot encoder
        """

        features = parse_single_example(serialized_example,
                                        self.get_template())
        image = decode_raw(features['image'], 'uint8')
        return image, one_hot(features['label'], self.classes)

    def normalize(self, image, label):
        """ Used in Validation Pipeline in order to evaluate the
        model without data augmentation

        :param image: tensor sample
        :param label: one hot label
        :return: 0-1 tensor and one-hot label
        """
        image = reshape(image, self.shape + (3, ))
        return cast(image, dtype='float32')/255., label

    def data_augmentation(self, image, label, th=.5):
        """  Basic Data augmentation  used in this pipeline
        in this part we only use rotate, flip and central crop

        :param image: tensor sample
        :param label: one hot label
        :param th: random value which triggered a augmentation

        :return: 0-1 tensor and one-hot label

        """
        image = reshape(image, self.shape + (3, ))
        if random() > th:
            image = central_crop(image, central_fraction=random())
            image = resize(image, self.shape, antialias=True)
        if random() > th:
            image = flip_left_right(image)
        if random() > th:
            image = rot90(image)
        return cast(image, dtype='float32')/255., label

    def __call__(self):
        self.build_train()
        self.build_test()


def _int64_feature(value):
    return Feature(int64_list=Int64List(value=[value]))


def _bytes_feature(value):
    return Feature(bytes_list=BytesList(value=[value]))


def _float_feature(value):
    return Feature(float_list=FloatList(value=[value]))
