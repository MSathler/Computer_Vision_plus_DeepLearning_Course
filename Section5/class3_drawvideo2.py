import cv2
import time
cap = cv2.VideoCapture(0)

time.sleep(1/2)

## CALLBACK FUNCTION RECTANGLE
def draw_rectangle(even):

## GLOBAL VARIABLES

while True:

    ret,frame = cap.read()

    ## DRAWING ON THE FRAME BASED OFF THE GLOBAL

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()