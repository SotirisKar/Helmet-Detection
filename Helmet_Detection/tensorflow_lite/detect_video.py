import tflite_runtime.interpreter as tflite
from collections import Counter
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='~/saved_models/model_edgetpu.tflite')
parser.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='~/saved_models/labels.txt')
parser.add_argument('--video', help='Name of the video to perform detection on', default='~/test_data/moto_vid_2.mp4')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
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
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]   # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects
    object_list = []
    objects = dict()
    
    for i in range(len(scores)):
        if ((scores[i] > MIN_THRESH) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * video_height)))
            xmin = int(max(1,(boxes[i][1] * video_width)))
            ymax = int(min(video_height,(boxes[i][2] * video_height)))
            xmax = int(min(video_width,(boxes[i][3] * video_width)))
            
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))                          # Example: helmet: 86%
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Font size
            label_ymin = max(ymin, labelSize[1])                                            # Draw label close to top of window
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
    cv2.imshow('Object Detector', cv2.resize(frame, (800, 600)))

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()


