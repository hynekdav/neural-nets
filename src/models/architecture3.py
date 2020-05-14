from models.architecture import Architecture
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU


class Architecture3(Architecture):
    def __init__(self, num_of_boxes=1):
        super().__init__(num_of_boxes)

    def build(self, input_shape=(256, 256, 3), optimizer='adam', activation='tanh'):
        alpha = 0.1

        model = Sequential()

        model.add(Conv2D(16, 3, strides=(1, 1), padding='same', input_shape=input_shape))
        model.add(LeakyReLU(alpha))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(32, 3, strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, 3, strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, 3, strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, 3, strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, 3, strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(1024, 3, strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha))
        model.add(Conv2D(1024, 3, strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha))
        model.add(Conv2D(1024, 1, strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(256))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_of_boxes * 4, activation=activation))

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model
