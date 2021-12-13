# Helmet Detection

<p align="center">
  <img width="1000" height="270" src="https://user-images.githubusercontent.com/36128807/145789823-586c4b98-a606-456a-a515-eae0b88f78cb.jpeg">
</p>

## About MHVision


[![](https://img.shields.io/badge/Visit-inAccel-darkblue)](https://inaccel.com/)
[![](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/)
[![](https://img.shields.io/badge/Tensorflow-2.7.0-orange)](https://www.tensorflow.org/)
[![](https://img.shields.io/badge/NumPy-1.21.1-lightblue)](https://numpy.org/)
[![](https://img.shields.io/badge/Pandas-1.3.2-darkblue)](https://pandas.pydata.org/)
[![](https://img.shields.io/badge/OpenCV-4.5.4-brightgreen)](https://opencv.org/)
[![](https://img.shields.io/badge/Pillow-8.3.2-9cf)](https://pillow.readthedocs.io/en/stable/)

A Tensorflow project that detects if the motorcycle rider wears helmet or not. Can be used either from raspberry pi 4 or Linux machine. Created with Python.

## How to Install requirements

Run the commands:

```sh
cd Helmet_Detection_w_RGB/utils
bash requirements.sh
```

or:

```sh
cd Helmet_Detection/tensorflow_lite
bash requirements.sh
```

## Run the detection

Run the commands:

```python
cd Helmet_Detection_w_RGB
python3 detect_camera.py
```

or without RGB:

```python
cd Helmet_Detection/tensorflow_lite
python3 detect_camera.py
```
