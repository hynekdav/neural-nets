from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import os
from keras.preprocessing.image import img_to_array
from preprocessing.preprocessing import Preprocesser


class SlidingWindowTrainer(object):
    def __init__(self, model):
        self.model = model

    def _preprocess_image(self, image):
        arr = img_to_array(image)
        return Preprocesser.normalize(arr)

    def train(self, training_data_folder, validation_data_folder, output_model_save_path):
        img_width, img_height = 64, 64
        batch_size = 4

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            preprocessing_function=self._preprocess_image)

        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            preprocessing_function=self._preprocess_image)

        train_generator = train_datagen.flow_from_directory(
            training_data_folder,
            target_size=(img_width, img_height),
            batch_size=batch_size)

        validation_generator = test_datagen.flow_from_directory(
            validation_data_folder,
            target_size=(img_width, img_height),
            batch_size=batch_size)

        samples_per_epoch = sum([len(files) for _, __, files in os.walk(training_data_folder)])
        validation_samples = sum([len(files) for _, __, files in os.walk(validation_data_folder)])
        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=samples_per_epoch // batch_size,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=validation_samples // batch_size,
            callbacks=[EarlyStopping(patience=3)],
            verbose=2)

        self.model.save(output_model_save_path)
