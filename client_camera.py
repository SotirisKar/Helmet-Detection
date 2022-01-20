import tflite_runtime.interpreter as tflite
from collections import Counter
from threading import Thread
import numpy as np
import argparse
import socket
import time
import cv2
import os

# Connect with Raspberry pi 4
def sendMessage(msg):
    s = socket.socket()        
    port = 8080             
    s.connect(('192.168.1.24', port))
    s.send(msg.encode())
    s.close()

class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        for i in range(1,6):
            if i==5:
                print('\nNo camera detected')
                os._exit(0)
            if not self.stream.isOpened():
                self.stream = cv2.VideoCapture(i)
                continue
            else:
                break
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
                    default=0.5)
parser.add_argument('--resolution', help='Camera resolution. Needs to be supported', default='640x480')
                    
args = parser.parse_args()
MODEL_PATH = 'models/model_edgetpu.tflite'
LABEL_PATH = 'models/labels.txt'
MIN_THRESH = float(args.thres_heightold)

res_width, res_height = args.resolution.split('x')
video_width, video_height = int(res_width), int(res_height)

# Load the model
interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]) #remove exp_delegetes if not Google Coral

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

# Videostream
videostream = VideoStream(resolution=(video_width,video_height),framerate=30).start()

while True:
    current_count=0
    t1 = cv2.getTickCount()
    frame_read = videostream.read()
    frame = frame_read.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'],input_data)
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
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine), (0, 255, 0), cv2.FILLED)
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 0), 1)
            else:
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 1)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                
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
        
    # Delay for the next 30 frames
    if x == 'No Helmet':
        delay_no = fps_count + 30
    elif x == 'Wears Helmet':
        delay_yes = fps_count + 30

    if fps_count <= delay_yes:
        x = 'Wears Helmet'
    elif fps_count <= delay_no:
        x = 'No Helmet'
    if fps_count > delay_yes and fps_count > delay_no:
        x = None

    # Draw Results & Send them to Raspberry Pi 4
    if x == 'Wears Helmet':
        cv2.putText(frame,x,(15,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        msg = 'H'
        try:
            sendMessage(msg)
        except Exception as e:
            print(e)
            os._exit(0)
    elif x == 'No Helmet':
        cv2.putText(frame,x,(15,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        msg = 'NH'
        try:
            sendMessage(msg)
        except Exception as e:
            print(e)
            os._exit(0)
    else:
        msg = 'N'
        try:
            sendMessage(msg)
        except Exception as e:
            print(e)
            os._exit(0)
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    if len(objects) != 0:
        cv2.putText(frame, str(objects),(15,115),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    cv2.imshow('Object Detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
os._exit(0)
