from flask import Flask, render_template, Response
import tflite_runtime.interpreter as tflite
from collections import Counter
from threading import Thread
from periphery import GPIO
from pytz import utc
import pandas as pd
import numpy as np
import argparse
import datetime
import cv2
import os

app = Flask(__name__)

# Read GPIOs
GPIO_29 = GPIO("/dev/gpiochip0", 7, "out")
GPIO_31 = GPIO("/dev/gpiochip0", 8, "out")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class VideoStream:
    def __init__(self,resolution=(1920, 1080),framerate=60):
        self.stream = cv2.VideoCapture(1)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        
parser = argparse.ArgumentParser()
parser.add_argument('--thres_heightold', help='Minimum confidence threshold',
                    default=0.4)
parser.add_argument('--resolution', help='Camera resolution. Needs to be supported', default='1920x1080')
                    
args = parser.parse_args()
MODEL_PATH = 'models/model_edgetpu.tflite'
LABEL_PATH = 'models/labels.txt'
MIN_THRESH = float(args.thres_heightold)

res_width, res_height = args.resolution.split('x')
video_width, video_height = int(res_width), int(res_height)

# Load the model
interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

# Load the labels
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Run videostream
videostream = VideoStream(resolution=(video_width,video_height),framerate=60).start()

def gen_frames(videostream, frame_rate_calc, freq, video_width, video_height):
    while True:
        current_count=0
        t1 = cv2.getTickCount()
        frame = videostream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])    
        interpreter.allocate_tensors()
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
        object_list = []
        objects = dict()

        for i in range(len(scores)):
            if ((scores[i] > MIN_THRESH) and (scores[i] <= 1.0)):
                ymin = int(max(1,(boxes[i][0] * video_height)))
                xmin = int(max(1,(boxes[i][1] * video_width)))
                ymax = int(min(video_height,(boxes[i][2] * video_height)))
                xmax = int(min(video_width,(boxes[i][3] * video_width)))
                
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                current_count+=1
                
                # Change Colors
                if object_name == 'helmet':
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine), (0, 160, 0), cv2.FILLED)
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 160, 0), 1)
                elif object_name == 'motorcycle':
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 160), 1)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine), (0, 0, 160), cv2.FILLED)
                else:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (160, 0, 0), 1)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine), (160, 0, 0), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    
                object_list.append(i)
                object_list[i] = object_name
                objects = Counter(object_list)
                
        x = None
        if 'motorcycle' in object_list:
            if object_list.count('motorcycle') == object_list.count('helmet'):
                x = 'Wears Helmet'
            elif object_list.count('motorcycle') > object_list.count('helmet'):
                x = 'No Helmet'

        # Start frame counter
        if 'fps_count' in locals():
            fps_count += 1
        else:
            fps_count = 0
            delay_no = 0
            delay_yes = 0
            
        if 'motocount' not in locals():
            motocount = 0
        if 'helmetcount' not in locals():
            helmetcount = 0
        if 'total_motos' not in locals():
            total_motos = 0
        if 'total_helmets' not in locals():
            total_helmets = 0

        # Delay for the next 20 frames
        if x == 'No Helmet':
            delay_no = fps_count + 20
        elif x == 'Wears Helmet':
            delay_yes = fps_count + 20

        if fps_count <= delay_yes:
            x = 'Wears Helmet'
        elif fps_count <= delay_no:
            x = 'No Helmet'
        if fps_count > delay_yes and fps_count > delay_no:
            x = None
            delay_no = 0
            delay_yes = 0
            fps_count = 0

        if x == 'Wears Helmet':
            cv2.rectangle(frame, (10,70),(170,97),(0,0,0),-1)
            cv2.putText(frame,x,(15,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1,cv2.LINE_AA)
            try:
                GPIO_29.write(True)
                GPIO_31.write(True)
            except Exception as e:
                print(e)
                os._exit(0)
        elif x == 'No Helmet':
            cv2.rectangle(frame, (10,70),(135,97),(0,0,0),-1)
            cv2.putText(frame,x,(15,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1,cv2.LINE_AA)
            try:
                GPIO_29.write(True)
                GPIO_31.write(False)
            except Exception as e:
                print(e)
                os._exit(0)
        else:
            try:
                GPIO_29.write(False)
                GPIO_31.write(False)
            except Exception as e:
                print(e)
                os._exit(0)
        cv2.rectangle(frame, (10,9),(135,37),(0,0,0),-1)
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1,cv2.LINE_AA)
        if len(objects) != 0:
            cv2.rectangle(frame, (10,35),(440,62+7),(0,0,0),-1)
            cv2.putText(frame, str(objects),(15,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1,cv2.LINE_AA)

        # Show camera feed to flask web server
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1   
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(videostream, frame_rate_calc, freq, video_width, video_height), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port='8080')
