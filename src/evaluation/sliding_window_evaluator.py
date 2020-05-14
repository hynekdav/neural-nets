from keras.preprocessing.image import img_to_array
from preprocessing.preprocessing import Preprocesser
from keras.preprocessing.image import ImageDataGenerator


class SlidingWindowEvaluator(object):
    def __init__(self, model):
        self.model = model

    def _preprocess_image(self, image):
        arr = img_to_array(image)
        return Preprocesser.normalize(arr)

    def evaluate(self, data_folder):
        eval_datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=self._preprocess_image)
        eval_data_generator = eval_datagen.flow_from_directory(data_folder, target_size=(64, 64), batch_size=4)

        return self.model.evaluate_generator(eval_data_generator)
