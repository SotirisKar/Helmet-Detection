from utils.display import displayMatrix
from multiprocessing import Process
import RPi.GPIO as GPIO
import time
import os

image = "images/connected.jpg"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
n=h=b=nh = Process(target=displayMatrix, args=[image])

GPIO.setwarnings(False)                                     # Disable Warnings
GPIO.setmode(GPIO.BCM)                                      # Read GPIOs
GPIO.setup(18, GPIO.IN)
GPIO.setup(24, GPIO.IN)
prev_gpio = [GPIO.input(18),GPIO.input(24)]                 # Remember previous state for GPIOs
print("INFO: Starting GPIO input detection.")

n.start()                                                   # Visual Confirmation that script runs
time.sleep(3)
n.terminate()

# To Get Rid of zombie colored pixels
image = "images/black.jpg"
b = Process(target=displayMatrix, args=[image])
b.start()
time.sleep(.5)
b.terminate()

while True:
    time.sleep(.5)
    if prev_gpio != [GPIO.input(18),GPIO.input(24)]:
        prev_gpio = [GPIO.input(18),GPIO.input(24)]
        if prev_gpio == [1,1]:
            image = "images/helmet.jpg"                     # Display helmet image
            h = Process(target=displayMatrix, args=[image])
            if nh.is_alive() == True:
                nh.terminate()                              # Kill previous image
            h.start()
        elif prev_gpio == [1,0]:                            # Display No Helmet image
            image = "images/no_helmet.jpg"
            nh = Process(target=displayMatrix, args=[image])
            if h.is_alive() == True:
                h.terminate()
            nh.start()
        elif prev_gpio == [0,0]:                            # Stop Displaying image
            if h.is_alive() == True:
                h.terminate()
            if nh.is_alive() == True:
                nh.terminate()

            b = Process(target=displayMatrix, args=[image])
            b.start()
            time.sleep(.5)
            b.terminate()
