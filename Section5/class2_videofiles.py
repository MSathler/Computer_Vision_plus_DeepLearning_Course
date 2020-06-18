import cv2
import time

cap = cv2.VideoCapture('video.mp4')

if cap.isOpened() == False:
    print('ERROS FILE NOT FOUND RO WRONG CODEC USED!')

while cap.isOpened():
    ret,frame = cap.read()

    if ret == True:

        #WRITER 20 FPS
        time.sleep(1/20) # for human see
        cv2.imshow('frame',frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    else:
        break
cap.release()
cv2.destroyAllWindows()