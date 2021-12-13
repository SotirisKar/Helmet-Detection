import tflite_runtime.interpreter as tflite
from collections import Counter
from threading import Thread
import numpy as np
import argparse
import cv2

class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
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
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='saved_models/model.tflite')
parser.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='saved_models/labels.txt')
parser.add_argument('--thres_heightold', help='Minimum confidence thres_heightold for displaying detected objects',
                    default=0.4)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='600x300')
                    
args = parser.parse_args()
MODEL_PATH = args.model
LABEL_PATH = args.labels
MIN_THRESH = float(args.thres_heightold)

res_width, res_height = args.resolution.split('x')
video_width, video_height = int(res_width), int(res_height)

# Load the model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)

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
#time.sleep(1)

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
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            current_count+=1
            
            # Change Colors
            if object_name == 'helmet':
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine), (0, 255, 0), cv2.FILLED) # Text Fill
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 0), 1)                                                              # Box
            else:
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 1)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1) # Text
                
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
        
    # Delay for the next 30 or 50 frames
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

    # Draw the Results
    if x == 'Wears Helmet':
        cv2.putText(frame,x,(15,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,x,(15,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
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
