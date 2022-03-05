#!/bin/bash
#bash requirements.sh

sudo apt install python3-dev python3-pip
pip3 install opencv-contrib-python
pip3 install APScheduler
pip3 install PyDrive
pip3 install pytz
pip3 install pandas
pip3 install numpy
pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_armv7l.whl#sha256=9175f1bb1c2f1f5c921117735a81943f85411248d78781453435a9bbfc212b91

echo "INFO: Installation Complete."
