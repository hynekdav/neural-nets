import numpy
from PIL import Image
from keras.preprocessing.image import img_to_array


class Preprocesser(object):
    @staticmethod
    def normalize(arr):
        arr = arr.astype('float')
        arr -= numpy.mean(arr)
        arr /= numpy.std(arr)
        return arr

    @staticmethod
    def preprocess_image(image):
        arr = img_to_array(image)
        img = Image.fromarray(Preprocesser.normalize(arr).astype('uint8'), 'RGB')
        return numpy.expand_dims(img, axis=0)[0]
