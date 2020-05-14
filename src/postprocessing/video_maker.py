from keras.models import load_model
import cv2
import time
from preprocessing.preprocessing import Preprocesser
from random import randint
import click
import numpy as np


class VideoMaker(object):
    @staticmethod
    def _process_predictions(predictions, img_size):
        x, w = predictions[0] * img_size[0], predictions[2] * img_size[0]
        y, h = predictions[1] * img_size[1], predictions[3] * img_size[1]
        return tuple(map(int, (max(x, 0), max(y, 0), min(x + w, img_size[0] - 1), min(y + h, img_size[0] - 1))))

    @staticmethod
    def annotate_video(in_video_path, out_video_path, models):
        split = lambda A, n=4: [A[i:i + n] for i in range(0, len(A), n)]

        models = [load_model(model) for model in models]

        reader = cv2.VideoCapture(in_video_path)
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(reader.get(cv2.CAP_PROP_FPS))

        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (width, height))

        colors = {model: (randint(0, 255), randint(0, 255), randint(0, 255)) for model in models}

        cnt = 0
        start = time.time()
        while reader.isOpened() and writer.isOpened():
            ret, frame = reader.read()
            cnt += 1
            if cnt % 100 == 0:
                click.echo('processed frame {0}'.format(cnt))
            if not ret:
                break
            if cnt % 4 != 0:
                writer.write(frame)
                continue
            resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            preprocessed = np.expand_dims(Preprocesser.preprocess_image(resized), axis=0)
            img = frame
            for model in models:
                predictions = model.predict(preprocessed)
                processed_pred = [VideoMaker._process_predictions(splited[0], (width, height)) for splited in
                                  split(predictions)]
                for (x, y, x1, y1) in processed_pred:
                    if any((x, y, x1, y1)):
                        img = cv2.rectangle(img, (x, y), (x1, y1), colors[model], 2)
            writer.write(img)
        end = time.time()
        return (end - start)
