import cv2 as cv 
import numpy as np 
import time

threshold = 0.4
supp_threshold = 0.3
YOLO_IMAGE_SIZE = 320
KNOWN_DISTANCE = 50 
CAR_WIDTH = 1.76
img_counter = 0

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

focal_car = focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, 2)

def findObject(mobject):
    class_id =[]
    conf_value = []
    box_location = []

    for obj in mobject:
        for prediction in obj:
            class_probabilities = prediction[5:]
            obj_id = np.argmax(class_probabilities)
            conf_ = class_probabilities[obj_id]

            if conf_ > threshold:
                class_id.append(obj_id)
                conf_value.append(float(conf_))
                w, h = int(prediction[2]*YOLO_IMAGE_SIZE), int(prediction[3]*YOLO_IMAGE_SIZE)
                x, y = int(prediction[0]*YOLO_IMAGE_SIZE-w/2), int(prediction[1]*YOLO_IMAGE_SIZE-h/2)
                box_location.append([x,y,w,h])
    
    predicted_box_id = cv.dnn.NMSBoxes(box_location, conf_value, threshold, supp_threshold)
    return predicted_box_id, box_location, class_id, conf_value

def show_detected_img(image, predicted_obj_id, bbox_location, class_ids, conf_values, width_ratio, height_ratio, focal_car, CAR_WIDTH, img_counter):
    count =0 
    for index in predicted_obj_id:
        boundingbox = bbox_location[index]
        x, y, w, h = int(boundingbox[0]*width_ratio), int(boundingbox[1]*height_ratio), int(boundingbox[2]*width_ratio), int(boundingbox[3]*height_ratio)
        
        if class_ids[index]==0:
            cv.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
            conff = 'PERSON' + str(int(conf_values[index]*100)) + '%'
            cv.putText(image, conff, (x,y-8), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,0,0), 1)
        if class_ids[index]==2:
            count = count+1 
            distance = distance_finder(focal_car, CAR_WIDTH, w)
            if distance < 0.6 and distance > 0.5:
                crop_img = image[y-5:y + h+5, x-5:x + w+5]
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv.imwrite(img_name, crop_img)
                img_counter += 1  
            conf = 'Distance:' + str(round(float(distance),2))
            cv.putText(image, conf, (x+30,y+20), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (100,255,200), 1)
            cv.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            conff = 'CAR' + str(int(conf_values[index]*100)) + '%'
            cv.putText(image, conff, (x,y-8), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,255), 1)
        
    return count, img_counter

classes = ['car', 'person']

model = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

capture = cv.VideoCapture('4.mp4')

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    
    org_width, org_height = frame.shape[1], frame.shape[0]
    blob = cv.dnn.blobFromImage(frame, 1/255, (320, 320), True, crop=False)
    model.setInput(blob)

    layers_name = model.getLayerNames()
    output_layers = [layers_name[index-1] for index in model.getUnconnectedOutLayers()]

    output = model.forward(output_layers)
    predicted_obj, box_location, class_id, conf_value = findObject(output)

    count, img_counter = show_detected_img(frame, predicted_obj, box_location, class_id, conf_value, org_width/YOLO_IMAGE_SIZE, org_height/YOLO_IMAGE_SIZE, focal_car, CAR_WIDTH, img_counter)
    counter = 'Number of Cars: ' + str(int(count))
    cv.putText(frame, counter, (10,20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)
    if count > 10:
        cv.putText(frame, "WARNING: Heavy Traffic", (10, 40), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    cv.imshow("Image", frame)
    cv.waitKey(1)

cv.destroyAllWindows()

