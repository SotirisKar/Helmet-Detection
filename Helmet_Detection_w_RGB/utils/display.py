#!/usr/bin/env python

from utils.rgbmatrix import RGBMatrix, RGBMatrixOptions
from PIL import Image
import time
import sys

def displayRGB(image):
    
    image = Image.open(image)

    # Configuration for the matrix
    options = RGBMatrixOptions()
    options.hardware_mapping = 'adafruit-hat'
    options.cols = 64
    options.rows = 32
    options.gpio_slowdown = 5
    options.pwm_bits = 1
    options.pwm_lsb_nanoseconds = 100
    matrix = RGBMatrix(options = options)

    # Make image fit our screen.
    image.thumbnail((matrix.width, matrix.height),Image.ANTIALIAS)
    matrix.SetImage(image.convert('RGB'))
    time.sleep(1.5)
