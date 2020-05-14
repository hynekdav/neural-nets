from models.architecture import Architecture
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.regularizers import l2


class Architecture2(Architecture):
    def __init__(self, num_of_boxes=1):
        super().__init__(num_of_boxes)

    def build(self, input_shape=(256, 256, 3), optimizer='adam', activation='tanh'):
        model = Sequential()

        model.add(Conv2D(64, 6, strides=(3, 3), kernel_regularizer=l2(0.01), input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(32, 6, strides=(4, 4), kernel_regularizer=l2(0.01), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(128, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_of_boxes * 4, activation=activation))

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        self.model = model

        return model
