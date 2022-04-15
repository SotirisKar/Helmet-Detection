from utils.display import displayMatrix
from multiprocessing import Process
import RPi.GPIO as GPIO
import time
import os

image = "images/connected.jpg"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
n=h=b=nh = Process(target=displayMatrix, args=[image])

GPIO.setwarnings(False)                                     # Disable GPIO Warnings
GPIO.setmode(GPIO.BCM)                                      # Read GPIOs
GPIO.setup(18, GPIO.IN)
GPIO.setup(24, GPIO.IN)
prev_gpio = [GPIO.input(18),GPIO.input(24)]                 # Remember previous state for GPIOs
print("INFO: Starting GPIO input detection.")

n.start()                                                   # Visual confirmation that script runs
time.sleep(3)
n.terminate()

image = "images/black.jpg"
b = Process(target=displayMatrix, args=[image])
b.start()                                                   # Get rid of zombie colored pixels
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
                nh.terminate()                              # Terminate previous image to display new
            h.start()
        elif prev_gpio == [1,0]:                            # Display no helmet image
            image = "images/no_helmet.jpg"
            nh = Process(target=displayMatrix, args=[image])
            if h.is_alive() == True:
                h.terminate()
            nh.start()
        elif prev_gpio == [0,0]:                            # Stop displaying any image
            if h.is_alive() == True:
                h.terminate()
            if nh.is_alive() == True:
                nh.terminate()

            b = Process(target=displayMatrix, args=[image])
            b.start()                                       # Get rid of zombie colored pixels
            time.sleep(.5)
            b.terminate()
