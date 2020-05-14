from models.architecture import Architecture
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Flatten, Dense, Conv2D, Dropout
from keras.regularizers import l2


class Architecture1(Architecture):
    def __init__(self, num_of_boxes=1):
        super().__init__(num_of_boxes)

    def build(self, input_shape=(256, 256, 3), optimizer='adam', activation='tanh'):
        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, strides=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, kernel_size=5, strides=5, padding="same", kernel_regularizer=l2(0.01)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_of_boxes * 4, activation=activation))

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        self.model = model

        return model
