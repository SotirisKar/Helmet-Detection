# SafeDisplay

<p align="center">
  <img width="100%" height="100%" src="https://user-images.githubusercontent.com/36128807/145789823-586c4b98-a606-456a-a515-eae0b88f78cb.jpeg">
</p>

[![](https://img.shields.io/badge/Visit-inAccel-darkblue)](https://inaccel.com/)
[![](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/)
[![](https://img.shields.io/badge/Tensorflow-2.7.0-orange)](https://www.tensorflow.org/)
[![](https://img.shields.io/badge/NumPy-1.21.1-lightblue)](https://numpy.org/)
[![](https://img.shields.io/badge/Pandas-1.3.2-darkblue)](https://pandas.pydata.org/)
[![](https://img.shields.io/badge/OpenCV-4.5.4-brightgreen)](https://opencv.org/)
[![](https://img.shields.io/badge/Pillow-8.3.2-9cf)](https://pillow.readthedocs.io/en/stable/)
[![](https://img.shields.io/badge/Flask-2.1.1-white)](https://flask.palletsprojects.com/)
[![](https://img.shields.io/badge/PyDrive-1.3.1-yellow)](https://pythonhosted.org/PyDrive/)
[![](https://img.shields.io/badge/APScheduler-3.9.1-yellow)](https://apscheduler.readthedocs.io/en/3.x/)

## About SafeDisplay

A Tensorflow project that detects if the motorcycle rider wears a helmet or not. Can be used either from raspberry pi 4 or Linux machine. 
Created with Python.

## How to Install requirements
For the Coral Dev Board installation,
Run the command:

```sh
sudo bash requirements.sh
```
For the Raspberry pi4 installation,
Run the command:

```sh
git clone https://github.com/hzeller/rpi-rgb-led-matrix.git
```
Now, move directory into the repo folder and install:

```sh
cd rpi-rgb-led-matrix
make build-python PYTHON=$(which python3)
sudo make install-python PYTHON=$(which python3)
```

After installation you need to move rgbmatrix folder to utils and delete rpi-rgb-led-matrix directory.
```sh
cd bindings/python
mv rgbmatrix /home/pi/SafeDisplay/utils
cd .. (Till you get to rpi directory)
rm -rf rpi-rgb-led-matrix
```

## Run SafeDisplay

For the Raspberry pi4 run the command:

```Python
sudo python3 display_input.py
```
For the Coral Dev Board run the command:

```Python
python3 camera_output.py
```
or for the web application:
```Python
python3 web-app.py
```
