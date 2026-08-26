import cv2 as cv
import numpy as np
from ultralytics import YOLO

def get_detections(model, image_path, is_thermal=False):
 if is_thermal:
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        image = np.squeeze(image) 
        #Removes single-dimensional entries from array shape (H, W, 1) into a 2D array of (H, W)).
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
 else:
        image = cv.imread(image_path)  # BGR by default
    

 
 results = model.predict(
    image,
    classes=[0],      # detect only person
    conf=0.30,        # confidence threshold below 30% will be removed
    iou=0.45,         # NMS threshold 
    imgsz=1280,       # image size
    verbose=False    
 ) #returns result as a list
 

 detections = []

 for box in results[0].boxes:
     cx, cy, w, h = box.xywh[0].tolist()  #This result is in cx, cy, w, h format

     x = cx - w / 2.0
     y = cy - h / 2.0

     boxes =(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

     conf_score = float(box.conf[0]) #Extract confidence of each box

     detections.append(
          {
               "box": boxes, 
               "confidence": conf_score ,
               "class_id": 0
          }
          
     )

 return detections

""" 
When you run: results = model.predict(image, ...)
I will get output like this: 
results
 ├── results[0]  → result for image 1
 ├── results[1]  → result for image 2
 └── results[2]  → result for image 3

 results[0] selects the result for first image the results are boxes, masks etc
 result[0].boxes select bounding box 
"""
