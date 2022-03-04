#!/bin/bash
#bash requirements.sh

sudo apt install python3-dev python3-pip
pip3 install opencv-python
pip3 install APScheduler
pip3 install PyDrive
pip3 install pytz
pip3 install pandas
pip3 install numpy
pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_armv7l.whl

echo "INFO: Installation Complete."
