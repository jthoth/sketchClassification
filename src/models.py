from tensorflow.keras import layers, Model, utils
from types import FunctionType

class BaseModelActions(Model):
    """ This class builds a dynamically models based
    homework instructions. The key idea was generate a list
    with a pre-defined flow.

    :param size: kernel size
    :param labels:# of classes
    :param k_init: How to initialize kernels or
    features maps

    """

    def __init__(self, size=(3, 3), labels=250, init='he_normal'):
        super(BaseModelActions, self).__init__()
        self.args = dict(kernel_initializer=init, padding='same')
        self.stride = dict(**dict(strides=(2, 2)), **self.args)
        self.pool = dict(pool_size=(3, 3), strides=2)
        self.size, self.labels = size, labels

    def build_parameters_on_model(self):
        """ Add dynamically values to class throughout
        setattr reserved word. This allows it Model tensorflow
        build the graph

        :param information_flow: list with steps procedures
        :return:
        """

        for i, layer in enumerate(self.information_flow):
            name = layer.__class__.__name__.lower()
            setattr(self, '{}_{}'.format(name, i), layer)

    def scale_reducer(self, _type, previous, fmap=64):
        """ Define is use a pool or Global Average as down
        scale

        :param pooling: features map
        :param previous: previous procedures added
        :return:
        """
        if _type == 'maxpool':
            previous += [layers.MaxPool2D(**self.pool)]
        elif _type == 'stride':
            args = [fmap,  self.size, self.stride]
            previous += [*self.common_block(*args)]
        elif _type == 'global':
            previous += [layers.GlobalAveragePooling2D()]

    def common_block(self, features, size, args):
        conv = layers.Conv2D(features, size, **args)
        batchnorm = layers.BatchNormalization()
        return [conv, batchnorm, layers.Activation('relu')]

    def build_block(self, features, previous, redux, num=4):
        """ Build the similarities defined in the homework graph

        :param features: features map
        :param previous: previous procedures added
        :param pool: down sacel using Maxpool or Global
        :param num: # of operations per block
        :return:
        """
        for feature_maps in features:
            args = [feature_maps,  self.size, self.args]
            for num_of_blocks in range(num):
                previous.extend(self.common_block(*args))
            self.scale_reducer(redux, previous, feature_maps)

class Vgg(BaseModelActions):
    """ This class builds a dynamically Vgg model described on
    homework instructions. The key idea was generate a list
    with a pre-defined flow.

    :param size: kernel size
    :param labels:# of classes
    :param init: How to initialize kernels or
    features maps
    """

    def __init__(self, size=(3, 3), labels=250, init='he_normal'):
        super(Vgg, self).__init__(size, labels, init)
        self.information_flow = self.information_process_flow()

    def information_process_flow(self):
        """Allow us to build a list of steps with similar block
        structure

        :return: list with operations procedures
        """
        steps = list()
        self.scale_reducer('stride', steps, fmap=64)
        self.build_block([64, 64, 128, 128], steps, 'maxpool')
        self.build_block([256], steps, 'global')
        steps += [layers.BatchNormalization()]
        return steps + [layers.Dense(self.labels, 'softmax')]

    def call(self, inputs, training=None, mask=None):
        """ Builds the transformation of VggModel given a list of
        procedures

        :param inputs: batch of samples
        :param training: define if the model is training
        of inferring
        :param mask:
        :return: transformed batched
        """
        x = inputs
        for i, layer in enumerate(self.information_flow):
            x = layer(x)
        return x


class Resnet(BaseModelActions):
    """ This class builds a dynamically Resnet model described on
    homework instructions. The key idea was generate a list
    with a pre-defined flow.

    :param size: kernel size
    :param labels:# of classes
    :param init: How to initialize kernels or
    features maps
    """

    def __init__(self, size=(3, 3), labels=250, init='he_normal'):
        super(Resnet, self).__init__(size, labels, init)
        self.information_flow = self.information_process_flow()
        self.skip_repair = self.compute_skip_repair()

    def skip_block(self, maps, down_first=True):
        args = self.stride if down_first else self.args
        steps = ['st', *self.common_block(maps, self.size, args)]
        steps += [layers.Conv2D(maps, self.size, **self.args)]
        steps += [layers.add, layers.BatchNormalization()]
        return steps + [layers.Activation('relu')]

    @staticmethod
    def walking_procedure():
        filters = [64, 64, 128, 128, 128, 128, 256, 256]
        scaling = [True, False] * (len(filters)//2)
        return [64, 64] + filters, [False, False] + scaling

    def compute_skip_repair(self):
        kw = dict(**dict(strides=(2, 2), **self.args))
        walker = self.walking_procedure()
        return [layers.Conv2D(maps, (1, 1), **kw)
               for maps, scale in zip(*walker) if scale]

    def information_process_flow(self):
        """Allow us to build a list of steps with similar block
        structure

        :return: list with operations procedures
        """
        steps = list()
        self.scale_reducer('stride', steps, fmap=64)
        for filters, scaled in zip(*self.walking_procedure()):
            steps.extend(self.skip_block(filters, scaled))
        self.scale_reducer('global', steps)
        steps += [layers.BatchNormalization()]
        return steps + [layers.Dense(self.labels, 'softmax')]

    def call(self, inputs, training=None, mask=None):
        """ Walk through resnet path constructed before

        :param inputs: batch of samples
        :param training: define if the model is training
        of inferring
        :param mask:
        :return: transformed batched
        """
        x, skip, fixers = inputs, None, 0
        for i, event in enumerate(self.information_flow):
            if type(event) == str:
                skip = x
            elif isinstance(event, FunctionType):
                if skip.shape[1:] != x.shape[1:]:
                    skip = self.skip_repair[fixers](skip)
                    fixers += 1
                x = event([skip, x])
            else:
                x = event(x)
        return x


class SqueezeExcitation(Resnet):
    """ This class builds a dynamically Resnet model described on
    homework instructions. The key idea was generate a list
    with a pre-defined flow.

    :param size: kernel size
    :param labels:# of classes
    :param init: How to initialize kernels or
    features maps
    """

    def __init__(self, size=(3, 3), labels=250, init='he_normal'):
        super(SqueezeExcitation, self).__init__(size, labels, init)
        self.squeeze_options = self.fill_squeze_procedures()

    def fill_squeze_procedures(self, ratio=4):
        steps, kw = [], dict(kernel_initializer='he_normal')
        filters, _ = self.walking_procedure()
        for filter_size in filters:
            steps.append([
                layers.GlobalAveragePooling2D(),
                layers.Dense(filter_size//ratio, **kw),
                layers.Activation('relu'),
                layers.Dense(filter_size, **kw),
                layers.Activation('sigmoid'),
                layers.multiply
            ])
        return steps

    def call(self, inputs, training=None, mask=None):
        """ Walk through resnet path constructed before

        :param inputs: batch of samples
        :param training: define if the model is training
        of inferring
        :param mask:
        :return: transformed batched
        """
        x, skip, fixers, squezer = inputs, None, 0, 0

        for i, event in enumerate(self.information_flow):
            if type(event) == str:
                skip = x
            elif isinstance(event, FunctionType):

                _input = x
                for sq in self.squeeze_options[squezer][:-1]:
                    x = sq(x)
                x = self.squeeze_options[squezer][-1]([_input, x])
                squezer += 1

                if skip.shape[1:] != x.shape[1:]:
                    skip = self.skip_repair[fixers](skip)
                    fixers += 1

                x = event([skip, x])
            else:
                x = event(x)
        return x
