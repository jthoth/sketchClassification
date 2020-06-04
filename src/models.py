from tensorflow.keras import layers, Model, utils


class VggModel(Model):
    """ This class builds a dynamically Vgg model described on
    homework instructions. The key idea was generate a list
    with a pre-defined flow.

    :param size: kernel size
    :param cls:# of classes
    :param k_init: How to initialize kernels or
    features maps

    """

    def __init__(self, size=(3, 3), cls=250, k_init='he_normal'):
        super(VggModel, self).__init__()
        init = dict(kernel_initializer=k_init, padding='same',
                    activation='relu')
        redux = dict(**dict(strides=(2, 2)), **init)
        self.information_flow = self.flow(size, cls, init, redux)
        self.build_graph( self.information_flow)

    def build_graph(self, information_flow):
        """ Add dynamically values to VggModel class throughout
        setattr reserved word. This allows it Model tensorflow
        build the graph

        :param information_flow: list with steps procedures
        :return:
        """
        for i, layer in enumerate(information_flow):
            name = layer.__class__.__name__.lower()
            setattr(self, '{}_{}'.format(name, i), layer)

    def flow(self, size, classes, kw, redux):
        """
        Allow us to build an arithmetic list building with
        similar block structure

        :param size: size of the kernels
        :param classes: num of classes in the output
        :param kw: random init
        :param redux: random init and stride 2
        :return: list with operations procedures
        """

        steps = [layers.Conv2D(64, size, **redux)]
        self.common_blocks(steps, [64, 64, 128, 128], size, kw)
        steps += [layers.Conv2D(256, size, **kw) for
                  _ in range(4)]
        steps += [layers.GlobalAveragePooling2D()]
        return steps + [layers.Dense(classes)]

    @staticmethod
    def common_blocks(steps, maps, size, kw, blocks=4):
        """ Build the similarities defined in the homework graph

        :param steps: previous steps
        :param maps: features maps
        :param size: kernel size
        :param kw: initialization and padding
        :param blocks: # of operations per block
        :return:
        """
        for f_map in maps:
            steps += [layers.Conv2D(f_map, size, **kw)
                      for _ in range(blocks)]
            steps += [layers.MaxPool2D(size, strides=2)]

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
            name = layer.__class__.__name__.lower()
            x = getattr(self, '{}_{}'.format(name, i))(x)
        return x


class ResNet(Model):
    """ This class builds a dynamically Resnet model described on
    homework instructions.

    :param size: kernel size
    :param classes:# of classes
    :param k_init: How to initialize kernels or
    features maps

    """

    def __init__(self,  size=(3, 3), cls=250, k_init='he_normal'):
        super(ResNet, self).__init__()