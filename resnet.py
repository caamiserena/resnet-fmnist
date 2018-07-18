import tensorflow as tf

from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Flatten, Dense
from keras.layers import add
from keras.models import Model

from keras import backend as K

BN_AXIS = 3 # K.image_data_format() == 'channels_last'

class Blocks():

    def __init__(self):
        return

    def BasicBlock(self, input_tensor, filters, downsample=False, identity=True, first_block=False):

        stride = 2
        if first_block:
            stride = 1

        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=stride, padding='same')(input_tensor)
        x = BatchNormalization(axis=BN_AXIS)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization(axis=BN_AXIS)(x)
        x = Activation('relu')(x)

        shortcut = input_tensor
        if not first_block:
            shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=stride, padding='same')(input_tensor)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x


    def BottleneckBlock(self, input_tensor, filters, downsample=True, identity=True):

        x = Conv2D(filters, (1, 1))(input_tensor)
        x = BatchNormalization(axis=BN_AXIS)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=BN_AXIS)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (1, 1))(x)
        x = BatchNormalization(axis=BN_AXIS)(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x


class ResNet():

    def model(self, input_size=(28,28,1), block='basic', layers=[2,2,2,2], filters=[64,128,256,512], num_classes=10):

        inputs = Input(shape=input_size)

         # Comienzo modelo
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv1')(inputs)
        x = BatchNormalization(axis=BN_AXIS, name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)
        # x = MaxPooling2D((3, 3), strides=(2, 2), name='conv1_maxpool')(x) Imagenes muy pequeñas, no es necesario.

        # Bloques residuales
        x = Blocks().BasicBlock(x, 32, first_block=True)
        x = Blocks().BasicBlock(x, 64, first_block=True)
        x = Blocks().BasicBlock(x, 128)

        # Final modelo
        x = AveragePooling2D((3, 3), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc10')(x)

        model = Model(inputs, x, name='resnet')

        return model
