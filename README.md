# Drone tracking in video stream using neural networks

This repository shows the software part of my bachelor's thesis created at FIT CTU in Prague in 2018.
The tesis (in Czech) can be viewed [here](https://dspace.cvut.cz/bitstream/handle/10467/76644/F8-BP-2018-Davidek-Hynek-thesis.pdf?sequence=-1&isAllowed=y).

## Contents

* __bin__ - this directory contains 3 of my trained models
* __src__ - in this directory there is the support application for the drone tracking neural networks training and evaluation as well as creation of dataset

### Requirements

* Python 3 - works well with version 3.5.2

### Using
First install all the requirements using:

```
$ pip install -r requirements.txt
```
Then run __drone_framework.py__ (with _--help_ parameter to see what are the options).

## Used software

* [Python](https://www.python.org/)
* [Keras](https://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [click](http://click.pocoo.org/6/)
* [OpenCV](https://opencv.org/)
* [Pillow](https://pillow.readthedocs.io/en/5.1.x/)

## Author

* **Hynek Davídek**
