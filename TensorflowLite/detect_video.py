import tflite_runtime.interpreter as tflite
from collections import Counter
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='models/model.tflite')
parser.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='models/labels.txt')
parser.add_argument('--video', help='Name of the video to perform detection on')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.4)
                    
args = parser.parse_args()

MODEL_PATH = args.model
LABEL_PATH = args.labels
VIDEO_PATH = args.video
MIN_THRESH = float(args.threshold)

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

# Calculate frame rate
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video
video = cv2.VideoCapture(VIDEO_PATH)
video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Start timer to calculate frame rate
while(video.isOpened()):
    current_count=0
    t1 = cv2.getTickCount()
    ret, frame1 = video.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
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

            if object_name == 'helmet':
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)

            object_list.append(i)
            object_list[i] = object_name
            objects = Counter(object_list)
            if object_list.count('helmet') > 0 and object_list.count('motorcycle') > 0:
                cv2.putText(frame,'Helmet',(190,160),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            elif object_list.count('motorcycle') > 0 and object_list.count('helmet') == 0:
                cv2.putText (frame, 'Motorcycle',(15,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    cv2.putText (frame,'Total Detection Count : ' + str(current_count),(15,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    if len(objects) != 0:
        cv2.putText(frame, str(objects),(15,115),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,55),2,cv2.LINE_AA)
    cv2.imshow('Object Detector', cv2.resize(frame, (800, 600)))

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
print("Exiting program ..")
