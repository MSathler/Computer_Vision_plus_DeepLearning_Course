import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
nadia = cv2.imread('../Computer-Vision-with-Python/DATA/Nadia_Murad.jpg',0)
denis = cv2.imread('../Computer-Vision-with-Python/DATA/Denis_Mukwege.jpg',0)
solvay = cv2.imread('../Computer-Vision-with-Python/DATA/solvay_conference.jpg',0)

face_cascade = cv2.CascadeClassifier('../Computer-Vision-with-Python/DATA/haarcascades/haarcascade_frontalface_default.xml')

def detect_eyes(img):
    if img is None:
        print('a')
    face_img = img.copy()

    eyes_recs = eye_cascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5)

    for (x,y,w,h) in eyes_recs:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img

def adj_detect_face(img):
    face_img = img.copy()

    face_recs = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)

    for (x,y,w,h) in face_recs:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img
result = adj_detect_face(solvay)

eye_cascade = cv2.CascadeClassifier('../Computer-Vision-with-Python/DATA/haarcascades/haarcascade_eye.xml')
result = detect_eyes(solvay)

cap = cv2.VideoCapture(0)
time.sleep(1)
while True:
    ret,frame = cap.read()
    frame = adj_detect_face(frame)
    cv2.imshow('Video Face Detect',frame)
    plt.show()
    k = cv2.waitKey(1)
    if k ==27:
        break

cap.release()
cv2.destroyAllWindows()

#plt.imshow(result,'gray')
#plt.show()