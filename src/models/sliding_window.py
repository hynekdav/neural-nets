from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout


class SlidingWindow(object):
    @staticmethod
    def build(input_shape=(64, 64, 3), optimizer='adam', activation='softmax'):
        model = Sequential()
        model.add(Conv2D(32, 3, strides=1, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, 3, strides=1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, 3, strides=1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(2, activation=activation))

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
