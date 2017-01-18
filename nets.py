from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Dense,
    Activation,
    Flatten,
    Convolution2D,
    MaxPooling2D,
    merge)
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.layers.advanced_activations import ELU
from keras import backend as K


# Legacy: will be replaced by NetBuilder
def _make_cnn(in_shape, binary=True, optimizer='rmsprop'):
    '''
    Make the CNN Model
    Parameters:
    in_shape: the numpy shape of the input matrix
    binary: True if the ground truth labels are binary
    '''

    model = Sequential()
    model.add(BatchNormalization(input_shape=in_shape[1:]))
    model.add(
        Convolution2D(
            128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    # model.add(GaussianNoise(1))
    model.add(GaussianDropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(GaussianDropout(0.25))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(GaussianDropout(0.25))

    if not binary:
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=optimizer)
    else:
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
    return model


class NetBuilder(object):
    def __init__(self,
                 output_activation,
                 num_outputs,
                 block_type='basic',
                 nb_row=3,
                 nb_col=3,
                 nb_filter=64,
                 ndense=128,
                 dropout=0.5):
        self._output_activation = output_activation
        self._nout = num_outputs
        self._block_type = block_type
        self._nb_row = nb_row
        self._nb_col = nb_col
        self._nb_filter = nb_filter
        self._dropout = dropout
        self._ndense = ndense 
        self._handle_dim_ordering()

    def build(self, input_shape, repetitions=[1,]):
        if len(input_shape) != 3:
            raise Exception(
                "Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")
        x = input = Input(shape=input_shape)
        x = self._bn_relu_conv(
            nb_filter=self._nb_filter, subsample=(2, 2))(x)
        x = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), border_mode="same")(x)
        x = GaussianDropout(self._dropout)(x)

        nb_filter = self._nb_filter
        if self._block_type == 'residual':
            for i, r in enumerate(repetitions):
                x = self._residual_block(
                    self._basic_block,
                    nb_filter=self._nb_filter,
                    repetitions=r,
                    is_first_layer=i == 0)(x)
                nb_filter *= 2
        elif self._block_type == 'seqcnn':
            for i, r in enumerate(repetitions):
                x = self._bn_relu_conv(nb_filter)(x)
                x = MaxPooling2D(pool_size=(2, 2),
                                     dim_ordering=K.image_dim_ordering())(x)
                x = GaussianDropout(self._dropout)(x)
        else:
            raise Exception('The CNN Type %s is not defined. Valid types are basic/residual' % self._block_type)

        # Dense block
        x = Flatten()(x)
        x = Dense(output_dim=self._ndense, init="he_normal")(x)
        x = ELU()(x)
        x = Dense(output_dim=self._nout, init="he_normal",
                  activation=self._output_activation)(x)
        model = Model(input=input, output=x)
        return model

    def _handle_dim_ordering(self):
        if K.image_dim_ordering() == 'tf':
            self._ROW_AXIS = 1
            self._COL_AXIS = 2
            self._CHANNEL_AXIS = 3
        else:
            self._CHANNEL_AXIS = 1
            self._ROW_AXIS = 2
            self._COL_AXIS = 3

    # Basic 3 X 3 convolution blocks.
    # Use for resnet with layers <= 34
    # Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    def _basic_block(self, nb_filter, init_subsample=(1, 1)):
        def f(input):
            conv1 = self._bn_relu_conv(nb_filter, subsample=init_subsample)(input)
            residual = self._bn_relu_conv(nb_filter)(conv1)
            return self._shortcut(input, residual)

        return f

    # Builds a residual block with repeating bottleneck blocks.
    def _residual_block(self, block_function, nb_filter, repetitions, is_first_layer=False):
        def f(input):
            for i in range(repetitions):
                init_subsample = (1, 1)
                if i == 0 and not is_first_layer:
                    init_subsample = (2, 2)
                input = block_function(nb_filter=nb_filter, init_subsample=init_subsample)(input)
            return input

        return f

    # Helper to build a BN -> ELU -> conv block
    # This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    def _bn_relu_conv(self, nb_filter, subsample=(1, 1)):
        def f(input):
            norm = BatchNormalization(mode=0, axis=self._CHANNEL_AXIS)(input)
            activation = ELU()(norm)
            return Convolution2D(
                nb_filter=nb_filter,
                nb_row=self._nb_row,
                nb_col=self._nb_col,
                subsample=subsample,
                init="he_normal",
                border_mode="same")(activation)

        return f

    # Helper to build a conv -> BN -> ELU block
    def _conv_bn_relu(self, nb_filter, subsample=(1, 1)):
        def f(input):
            conv = Convolution2D(
                nb_filter=nb_filter,
                nb_row=self._nb_row,
                nb_col=self._nb_col,
                subsample=subsample,
                init="he_normal",
                border_mode="same")(input)
            norm = BatchNormalization(mode=0, axis=self._CHANNEL_AXIS)(conv)
            return ELU()(norm)

        return f

    # Adds a shortcut between input and residual block and merges them with "sum"
    def _shortcut(self, input, residual):
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        stride_width = input._keras_shape[self._ROW_AXIS] // residual._keras_shape[self._ROW_AXIS]
        stride_height = input._keras_shape[
            self._COL_AXIS] // residual._keras_shape[self._COL_AXIS]
        equal_channels = residual._keras_shape[self._CHANNEL_AXIS] == input._keras_shape[self._CHANNEL_AXIS]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Convolution2D(
                nb_filter=residual._keras_shape[self._CHANNEL_AXIS],
                nb_row=1,
                nb_col=1,
                subsample=(stride_width, stride_height),
                init="he_normal",
                border_mode="valid")(input)

        return merge([shortcut, residual], mode="sum")
