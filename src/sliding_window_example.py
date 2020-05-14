import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

from evaluation.sliding_window_utils import non_max_suppression, pyramid, sliding_window
from preprocessing.preprocessing import Preprocesser


def sliding_window_processer(model_path, image_path, threshold=0.7):
    model = load_model(model_path)
    image = load_img(image_path, target_size=(448, 448))
    (winW, winH) = (64, 64)

    arr = img_to_array(image)
    image = Image.fromarray(Preprocesser.normalize(arr).astype('uint8'), 'RGB')

    height, width = image.size
    possible_windows = list()
    for resized in pyramid(image, scale=1.5):
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            if window.size[0] != winH or window.size[1] != winW:
                continue
            preds = model.predict(np.expand_dims(window, axis=0))[0]
            if preds[0] >= threshold:
                possible_windows.append(
                    (x / width, y / height, min((x + winW) / width, 1), min((y + winH) / height, 1), preds[0]))
    possible_windows = non_max_suppression(np.asarray(possible_windows), 0.1)
    return possible_windows
