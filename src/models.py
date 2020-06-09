from tensorflow.keras import layers, Model, utils
from types import FunctionType


class BaseModelActions(Model):
    """ BaseModelActions is a class which contains some
    commons procedures related with homework architectures
    blocks

    :param size: kernel_size of the model
    :param labels: # of problem classes
    :param init: how the network params should begin
    """

    def __init__(self, size=(3, 3), labels=250, init='he_normal'):

        super(BaseModelActions, self).__init__()
        self.args = dict(kernel_initializer=init, padding='same')
        self.stride = dict(**dict(strides=(2, 2)), **self.args)
        self.pool = dict(pool_size=(3, 3), strides=2)
        self.size, self.labels = size, labels

    def scale_reducer(self, _type, previous, filters=64):
        """Contains multiples methods to allow us down size the
        image, to added a chain procedure

        :param _type:  define what kind of scale you need
        :param previous: is a chain container information flow
        :param filters: this is only used for convolution procedure
        :return:
        """
        if _type == 'maxpool':
            previous += [layers.MaxPool2D(**self.pool)]
        elif _type == 'stride':
            args = [filters,  self.size, self.stride]
            previous += [*self.common_block(*args)]
        elif _type == 'global':
            previous += [layers.GlobalAveragePooling2D()]

    @staticmethod
    def common_block(filters, size, args):
        """ This method contains the basic layer structure

        :param filters: # of features maps
        :param size: kernel_size to operate over image
        :param args: contains a dict which help to define the
        behaviour of convolution procedure
        :return: a list with procedures
        """
        convolve = layers.Conv2D(filters, size, **args)
        batch_norm = layers.BatchNormalization()
        return [convolve, batch_norm, layers.Activation('relu')]


class PlainNetwork(BaseModelActions):
    """ Allow us to build a plain neural network using a basic
    building block. This class is useful to build Vgg Models
    in a dynamic way. The most important building block is
    the information flow, which allow to define the steps

    :param size: kernel_size of the model
    :param labels: # of problem classes
    :param init: how the network params should begin
    """

    def __init__(self, size=(3, 3), labels=250, init='he_normal'):
        super(PlainNetwork, self).__init__(size, labels, init)
        self.information_flow = self.build_information_flow()

    def add_blocks(self, filters, previous, redux, blocks=4):
        """ Add dynamically operations over chain information
        flow in order to define the block path in the
        computational graph

        :param filters: # of features maps
        :param previous: is a chain container information flow
        which will be edited in this method adding new layers
        based on specifications
        :param redux: down scale operator type
        :param blocks: # of blocks per common filter
        :return:
        """
        for feature_maps in filters:
            args = [feature_maps,  self.size, self.args]
            for num_of_block in range(blocks):
                previous.extend(self.common_block(*args))
            self.scale_reducer(redux, previous, feature_maps)

    def build_information_flow(self):
        """ Add dynamically operations over chain information
        flow in order to define the steps of the  whole
        computational graph  procedure

        :return: list with all procedures which will be added
        to this model in order to register it in Model tensorflow
        class
        """
        steps = list()
        self.scale_reducer('stride', steps, filters=64)
        self.add_blocks([64, 64, 128, 128], steps, 'maxpool')
        self.add_blocks([256], steps, 'global')
        steps += [layers.BatchNormalization()]
        return steps + [layers.Dense(self.labels, 'softmax')]

    def call(self, inputs, training=None, mask=None):
        """This method define the computational graph path

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


class SkipNetwork(BaseModelActions):
    """ Generate a dynamically SkipNetwork  (a.k.a ResNet),
    the main idea was to build the   information flow like
    Plain Network described above and a skip repair procedure

    Additionally we introduce a string params called 'st'
    which means storage the input in that point.

    :param size: kernel size
    :param labels:# of classes
    :param init: How to initialize kernels or
    features maps
    """

    def __init__(self, size=(3, 3), labels=250, init='he_normal'):
        super(SkipNetwork, self).__init__(size, labels, init)
        self.information_flow = self.build_information_flow()
        self.skip_repair = self.build_skip_repair_steps()

    @staticmethod
    def walking_procedure():
        """
        This method define the proposed ResNet network
        architecture defined in the homework

        :return:
        """
        filters = [64, 64, 128, 128, 128, 128, 256, 256]
        scaling = [True, False] * (len(filters)//2)
        return [64, 64] + filters, [False, False] + scaling

    def skip_block(self, maps, down_first=True):
        """ This building block of ResNet architecture was inspired
        on Identity Mappings in Deep Residual Networks
        or a pre-activation variant of residual block
        arXiv:1603.05027v3,2016.

        :param maps:  # of features maps
        :param down_first:  define if exists a down scale in the
        first convolution layer
        :return:  a list with Skip block procedure

        """
        args = self.stride if down_first else self.args
        steps = [layers.BatchNormalization()]
        steps += [layers.Activation('relu')]
        steps += ['st', *self.common_block(maps, self.size, args)]
        steps += [layers.Conv2D(maps, self.size, **self.args)]
        steps += [layers.add, layers.BatchNormalization()]

        return steps + [layers.Activation('relu')]

    def build_skip_repair_steps(self):
        """ Based on the walking_procedure this method add
        a convolution of 1x1 in order to repair the dimensions

        :return: a list with the total number of repairs block

        """
        kw = dict(**dict(strides=(2, 2), **self.args))
        walker = self.walking_procedure()
        return [layers.Conv2D(maps, (1, 1), **kw)
                for maps, scale in zip(*walker) if scale]

    def build_information_flow(self):
        """ Add dynamically operations over chain information
        flow in order to define the steps of the  whole
        computational graph  procedure

        :return: list with all procedures which will be added
        to this model in order to register it in Model tensorflow
        class
        """
        steps = list()
        self.scale_reducer('stride', steps, filters=64)
        for filters, scaled in zip(*self.walking_procedure()):
            steps.extend(self.skip_block(filters, scaled))
        self.scale_reducer('global', steps)
        steps += [layers.BatchNormalization()]
        return steps + [layers.Dense(self.labels, 'softmax')]

    def call(self, inputs, training=None, mask=None):
        """This method define the computational graph path
        and it is called when a batch of images is passed
        through the network

        :param inputs: batch of samples
        :param training: define if the model is training
        of inferring
        :param mask:
        :return: transformed batched
        """

        x, skip, fixer = inputs, None, 0
        for i, event in enumerate(self.information_flow):
            if type(event) == str:
                skip = x  # when we to store the input
            elif isinstance(event, FunctionType):
                if skip.shape[1:] != x.shape[1:]:
                    skip = self.skip_repair[fixer](skip)
                    fixer += 1
                x = event([skip, x])  # add procedure
            else:
                x = event(x)  # rest ot the cases

        return x


class SqueezeExcitation(SkipNetwork):
    """ This class builds a dynamically SqueezeExcitation Net
     the model was described int the homework instructions.

     The key idea was generate a list  with a pre-defined flow
     based on the father of this class.

    :param size: kernel size
    :param labels:# of classes
    :param init: How to initialize kernels or
    features maps
    """

    def __init__(self, size=(3, 3), labels=250, init='he_normal'):
        super(SqueezeExcitation, self).__init__(size, labels, init)
        self.squeeze_options = self.fill_squeeze_procedures()

    def fill_squeeze_procedures(self, ratio=.25):
        """ Allow us to define the fire path in the walking
        procedure. The used patch was inspired in the original
        paper.

        :param ratio: quantity which will scale the channels
        of the images blocks

        :return:
        """
        steps, kw = [], dict(kernel_initializer='he_normal')
        filters, _ = self.walking_procedure()
        for filter_size in filters:
            steps.append([
                layers.GlobalAveragePooling2D(),
                layers.Dense(int(filter_size * ratio), **kw),
                layers.Activation('relu'),
                layers.Dense(filter_size, **kw),
                layers.Activation('sigmoid'),
                layers.multiply])

        return steps

    def call(self, inputs, training=None, mask=None):
        """This method define the computational graph path
        and it is called when a batch of images is passed
        through the network, here is applied the
        skip fixed and fire steps.

        :param inputs: batch of samples
        :param training: define if the model is training
        of inferring
        :param mask:
        :return: transformed batched
        """
        x, skip, fixers, sq = inputs, None, 0, 0

        for i, event in enumerate(self.information_flow):
            if type(event) == str:
                skip = x  # when we to store the input
            elif isinstance(event, FunctionType):
                # init squeeze operations
                _x = x
                for squeeze in self.squeeze_options[sq][:-1]:
                    x = squeeze(x)
                x = self.squeeze_options[sq][-1]([_x, x])
                # end the squeeze operations with multiplication
                sq += 1
                if skip.shape[1:] != x.shape[1:]:
                    skip = self.skip_repair[fixers](skip)
                    fixers += 1

                x = event([skip, x])  # add ResNet procedure
            else:
                x = event(x)  # rest ot the cases

        return x


MODELS = dict(Vgg=PlainNetwork)
MODELS.__setitem__('ResNet', SkipNetwork)
MODELS.__setitem__('SqueezeExcitation', SqueezeExcitation)