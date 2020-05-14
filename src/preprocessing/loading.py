import numpy as np
from .preprocessing import Preprocesser
import os
import imghdr
from keras.preprocessing.image import load_img


class DataLoader(object):
    @staticmethod
    def _parse_file(file_path):
        data = []
        with open(file_path) as in_file:
            for line in in_file:
                line = list(map(float, line.strip().split()))
                data.append(line[1:])
        return data

    @staticmethod
    def _parse(folder, num_of_boxes):
        parsed = []
        for root, _, files in os.walk(folder):
            for file in sorted(files):
                if file.endswith('txt'):
                    data = DataLoader._parse_file(os.path.join(root, file))
                    tmp_parsed = []
                    for box in data:
                        tmp_parsed.extend(box)
                    needed = 4 * num_of_boxes - len(tmp_parsed)
                    if needed > 0:
                        tmp_parsed.extend([0 for _ in range(needed)])
                    parsed.append(tmp_parsed)
        return np.asarray(parsed)

    @staticmethod
    def _load_images(folder):
        images = []
        for root, _, files in os.walk(folder):
            for file in sorted(files):
                if not imghdr.what(os.path.join(root, file)) is None:
                    image = Preprocesser.preprocess_image(load_img(os.path.join(root, file), target_size=(256, 256)))
                    images.append(image)
        return np.asarray(images)

    @staticmethod
    def load_data(folder, num_of_boxes=1):
        if num_of_boxes <= 0 or num_of_boxes > 5:
            raise ValueError('Can not load more than 5 boxes or less than 1.')
        images = DataLoader._load_images(folder)
        labels = DataLoader._parse(folder, num_of_boxes)
        return (images, labels)
