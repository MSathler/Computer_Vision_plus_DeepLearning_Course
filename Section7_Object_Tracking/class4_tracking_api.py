import cv2

def ask_for_tracker():
    print("Welcome! What Tracker API woul you like to use?")
    print("Enter 0 for BOOSTING: ")
    print("Enter 1 for MIL: ")
    print("Enter 2 for KCF: ")
    print("Enter 3 for TDL: ")
    print("Enter 4 for MEDIANFLOW: ")

    choise = input('Please select your tracker: ')

    if choise == '0':
        tracker = cv2.TrackerBoosting_create()
    if choise == '1':
        tracker = cv2.TrackerMIL_create()
    if choise == '2':
        tracker = cv2.TrackerKCF_create()
    if choise == '3':
        tracker = cv2.TrackerTDL_create()
    if choise == '4':
        tracker = cv2.TrackerMedianFlow_create()
    return tracker

tracker = ask_for_tracker()

tracker_name = (str(tracker).split()[0][1:])

# READ VIDEO
cap = cv2.VideoCapture(0)

# READ FIRST FRAME
ret,frame = cap.read()

roi = cv2.selectROI(frame,False)

ret = tracker.init(frame,roi)

while True:
    # READ A NEW FRAME
    ret,frame = cap.read()

    # UPDATE TRACKER
    success,roi = tracker.update(frame)

    #roi variable is a tuple of 4 floats
    (x,y,w,h) = tuple(map(int,roi))

    if success:
    # TRACKING SUCCESS
        p1 = (x,y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame,p1,p2,(0,255,0),3)
    else:
        #TRACKING FAILURE
        cv2.putText(frame, "Failure to Detect Tracking!!",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.putText(frame,tracker_name,(20,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.imshow(tracker_name,frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()