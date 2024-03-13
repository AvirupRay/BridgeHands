import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

capture = cv.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20
imgSize = 300

while True:
    isTrue, frame = capture.read()
    hands, img = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y , w, h = hand['bbox']

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255

        cropped_img = img[y - offset: y + h + offset, x: x + w + offset]

        imageCropShape = cropped_img.shape

        imgWhite[0:imageCropShape[0],0:imageCropShape[1]] = cropped_img

        aspectRatio = h/w

        if aspectRatio > 1:
            k= imgSize/h
            wCal=math.ceil(k*w)

        cv.imshow("ImageCrop", cropped_img)
        cv.imshow("ImageWhite", imgWhite)

    # cv.imshow("Hand Detection", cropped_img)
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()