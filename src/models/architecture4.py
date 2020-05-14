from models.architecture import Architecture
from keras.layers import Input, MaxPooling2D, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers.merge import concatenate
from keras.models import Model


class Architecture4(Architecture):
    def __init__(self, num_of_boxes=1):
        super().__init__(num_of_boxes)

    def build(self, input_shape=(256, 256, 3), optimizer='adam', activation='tanh'):
        input_layer = Input(shape=input_shape)

        squeeze = Conv2D(16, kernel_size=1, strides=2, padding='same', activation='relu')(input_layer)
        expand_1x1 = Conv2D(64, kernel_size=1, strides=2, padding='same', activation='relu')(squeeze)
        expend_3x3 = Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(squeeze)

        merged = concatenate([expand_1x1, expend_3x3])

        norm_1 = BatchNormalization()(merged)
        conv_3 = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(norm_1)
        conv_4 = Conv2D(16, kernel_size=3, strides=4, padding='same', activation='relu')(conv_3)
        max_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_4)

        flatten_1 = Flatten()(max_1)
        dense_1 = Dense(256, activation='relu')(flatten_1)
        dropout_1 = Dropout(0.5)(dense_1)
        out = Dense(self.num_of_boxes * 4, activation=activation)(dropout_1)

        model = Model(inputs=input_layer, outputs=out)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model
