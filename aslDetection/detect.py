import cv2 as cv
from cvzone.HandTrackingModule import HandDetector

capture = cv.VideoCapture(1)
detector = HandDetector(maxHands = 1)
while True:
    isTrue, frame = capture.read()
    hands, img = detector.findHands(frame)
    cv.imshow("Hand Detection", frame)
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()