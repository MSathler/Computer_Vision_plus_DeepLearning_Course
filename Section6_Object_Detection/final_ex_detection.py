import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()

def detect_and_blur_plate(img):
    #pass
    russian_plate_col_copy = img.copy()
    roi = img.copy()

    plate_recs = plate_cascade.detectMultiScale(russian_plate_col_copy, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in plate_recs:

        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi,7)

        russian_plate_col_copy[y:y+h,x:x+w] = blurred_roi

    return russian_plate_col_copy


def detect_plate(img):
    russian_plate_col_copy = img.copy()

    plate_recs = plate_cascade.detectMultiScale(russian_plate_col_copy,scaleFactor=1.2,minNeighbors=5)
    for (x,y,w,h) in plate_recs:
        cv2.rectangle(russian_plate_col_copy,(x,y),(x+w,y+h),(255,0,0),5)
    return russian_plate_col_copy

russian_plate = cv2.imread('../DATA/car_plate.jpg')
russian_plate_col = cv2.cvtColor(russian_plate,cv2.COLOR_BGR2RGB)
plate_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_russian_plate_number.xml')

result = detect_plate(russian_plate_col)
result2 = detect_and_blur_plate(russian_plate_col)
display(result2)