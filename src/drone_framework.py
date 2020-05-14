#!/usr/bin/env python3

import click
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model

from dataset_making.dataset_creator import DatasetCreator
from evaluation.architecture_evaluator import ArchitectureEvaluator
from evaluation.sliding_window_evaluator import SlidingWindowEvaluator
from preprocessing.loading import DataLoader
from postprocessing.video_maker import VideoMaker
from training.architecture_trainer import ArchitectureTrainer
from training.sliding_window_trainer import SlidingWindowTrainer

from models import *

from sliding_window_example import sliding_window_processer


def die(why):
    click.echo(why)
    exit(-1)


@click.group()
def cli():
    """
    Entry point for drone searching AI framework.
    """
    pass


@cli.command()
@click.option('-n', type=int, default=1, help='Every n-th frame to be saved.')
@click.argument('video_file', type=click.Path(exists=True))
@click.argument('out_folder', type=click.Path())
def create_dataset(n, video_file, out_folder):
    """
    Splits given VIDEO_FILE by every n-th frame and saves frames as JPEGs into OUT_FOLDER. VIDEO_FILE should be MP4.
    """
    if n < 1:
        die('Invalid number of n-th frame.')
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    created = DatasetCreator.create_dataset(video_file, out_folder, n)
    click.echo(
        'Created dataset of {0} images. Now you need to label it using your favorite image annotating tool.'.format(
            created))


@cli.command()
@click.option('--model_type', type=click.Choice(['arch1', 'arch2', 'arch3', 'arch4', 'sliding_window']),
              help='Architecture to train.')
@click.option('-n', type=int, default=1, help='Maximum number of drones detected in one frame (max. 5).')
@click.option('-v', type=float, default=0.1, help='Ratio of training to validation data (must be in interval (0, 1)).')
@click.argument('data_folder', type=click.Path(exists=True))
@click.argument('out_model_file', type=click.Path())
def train(model_type, n, v, data_folder, out_model_file):
    """
    Trains architecture (as listed in my thesis) to recognize maximum 'n' drones in image '256x256' pixels.
    Provided data in DATA_FOLDER must be in format 'file1.jpg file1.txt file2.jpg file2.txt' and so on.
    Text files are expected to contain coordinates as per YOLO format.
    Trained model is stored as H5 file to OUT_MODEL_FILE.
    Data for sliding window classifier must be in format folder/training and folder/validation.
    """
    if n < 0 or n > 5:
        die('Invalid number of possible drones to detect entered.')
    if v <= 0 or v >= 1:
        die('Invalid ratio of training to validation data entered.')
    if os.path.isdir(out_model_file):
        die('Output model file can\'t be a folder.')
    if model_type == 'sliding_window':
        trainer = SlidingWindowTrainer(sliding_window.SlidingWindow.build())
        trainer.train(os.path.join(data_folder, 'train'), os.path.join(data_folder, 'validation'), out_model_file)
    else:
        if model_type == 'arch1':
            arch = architecture1.Architecture1(n)
        elif model_type == 'arch2':
            arch = architecture2.Architecture2(n)
        elif model_type == 'arch3':
            arch = architecture3.Architecture3(n)
        else:
            arch = architecture4.Architecture4(n)
        trainer = ArchitectureTrainer(arch.build(), n)
        trainer.train(data_folder, out_model_file)
    click.echo(
        'Model successfully trained and saved. Don\'t freak out if there is a Tensorflow exception, it is known bug in Tensorflow.')


@cli.command()
@click.option('-n', type=int, default=0, help='Maximum number of drones detectable in one frame (max. 5).')
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_folder', type=click.Path(exists=True))
def evaluate(n, model_path, data_folder):
    """
    Evaluates given model (H5 file in MODEL_PATH) recognizing up-to 'n' drones in video.
    Will evaluate on data in DATA_FOLDER.
    If model is sliding window, then 'n' will be ignored.
    If 'n' is 0, then script will suppose that input model is classifier for sliding window and will evaluate its accuracy.
    """
    if n < 0 or n > 5:
        die('Invalid number of possible drones to detect entered.')
    if os.path.isdir(model_path):
        die('Model path can not be a folder.')
    model = load_model(model_path)
    if n > 0:
        evaluator = ArchitectureEvaluator(model, n)
        images, boxes = DataLoader.load_data(data_folder, n)
        ious = evaluator.evaluate(images, boxes)
        mean, var, mn, mx = np.mean(ious), np.var(ious), np.amin(ious), np.amax(ious)
        click.echo(
            'Model evaluated with {0:.4f} mean IoU and {1:.4f} IoU variance. Min IoU is {2:.4f}, max is {3:.4f}.'.format(
                mean, var, mn, mx))
    else:
        evaluator = SlidingWindowEvaluator(model)
        accuracy = evaluator.evaluate(data_folder)[0]
        click.echo('Sliding window model evaluated with accuracy of {0:.2f}%.'.format(accuracy * 100))


@cli.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.argument('models', nargs=-1, type=click.Path(exists=True))
def make_video(input_video, output_video, models):
    """
    Creates an example video from INPUT_VIDEO saved as AVI into OUTPUT_VIDEO.
    The video will be processed by each model given in MODELS.
    """
    if os.path.isdir(output_video):
        die('Output video can\'t be a folder.')
    elapsed = VideoMaker.annotate_video(input_video, output_video, models)
    click.echo('Created video in {0:.1f} seconds.'.format(elapsed))


@cli.command()
@click.option('-t', default=0.7, type=float, help='Classifier threshold.')
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
def sliding_window_example(t, model_path, image_path):
    """
    Performs sliding window regression on image given in IMAGE_PATH.
    Classification is performed with model from MODEL_PATH.
    Option 't' allows to set own value of classification threshold.
    Returned predictions are in array format and contains [x1 y1 x2 y2 prob] where x1, y1, x2 and y2 are box coords
    and prob is probability of drone at position of the rectangle.
    """
    if t <= 0 or t >= 1:
        die('Threshold must be in range (0, 1).')
    if os.path.isdir(model_path):
        die('Model path can not be a folder.')
    if os.path.isdir(image_path):
        die('Image path can not be a folder.')
    results = sliding_window_processer(model_path, image_path, t)
    click.echo("Predicted rectangles:")
    for result in results:
        click.echo(result)


cli()
