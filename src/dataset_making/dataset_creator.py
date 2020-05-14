import cv2
import os
import click


class DatasetCreator(object):
    @staticmethod
    def create_dataset(video_file, out_folder, n=30):
        if not os.path.isdir(out_folder) or not os.path.exists(out_folder):
            os.mkdir(out_folder)
        video = cv2.VideoCapture(video_file)
        frame_counter, save_counter = 0, 0
        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            frame_counter += 1
            if frame_counter % n == 0:
                save_counter += 1
                click.echo('processed frame {0}'.format(frame_counter))
                cv2.imwrite(os.path.join(out_folder, 'image{0}.jpg'.format(save_counter)), frame)
        video.release()
        return save_counter
