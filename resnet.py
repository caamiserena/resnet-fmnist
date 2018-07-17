import tensorflow as tf

from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Flatten, Dense
from keras.models import Model

from keras import backend as K

BN_AXIS = 3 # K.image_data_format() == 'channels_last'

class BasicBlock():

    def __init__(self):
        return

class BottleneckBlock():

    def __init__(self):
        return


class ResNet():

    def resnet18(self, input_size=(28,28,1), block='basic', layers=[2,2,2,2], num_classes=10):

        inputs = Input(shape=input_size)

         # Comienzo modelo
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(inputs)
        x = BatchNormalization(axis=BN_AXIS, name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)
        # x = MaxPooling2D((3, 3), strides=(2, 2), name='conv1_maxpool')(x) Imagenes muy pequeñas, no es necesario.

        # Bloques residuales


        # Final modelo
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc10')(x)

        model = Model(inputs, x, name='resnet')

        return model
