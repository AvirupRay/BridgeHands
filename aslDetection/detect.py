import cv2 as cv
from cvzone.HandTrackingModule import HandDetector

capture = cv.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20

while True:
    isTrue, frame = capture.read()
    hands, img = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y , w, h = hand['bbox']
        cropped_img = img[y - offset: y + h + offset, x: x + w + offset]
        cv.imshow("Hand Detection", cropped_img)

    cv.imshow("Hand Detection", img)
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()