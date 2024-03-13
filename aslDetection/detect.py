import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from termcolor import colored


capture = cv.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20
img_size = 400

while True:
    isTrue, frame = capture.read()
    hands, img = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y , w, h = hand['bbox']
        white_img = np.ones((img_size, img_size, 3), np.uint8) * 255
        cropped_img = img[y - offset: y + h + offset, x - offset: x + w + offset]

        cropped_img_shape = cropped_img.shape
        print(colored(f"Height: {cropped_img_shape[0]}  Width: {cropped_img_shape[1]}", "blue"))

        white_img[0:cropped_img_shape[0], 0:cropped_img_shape[1]] = cropped_img

        cv.imshow("Hand Image", cropped_img)
        cv.imshow("White Hand Image", white_img)
    
    cv.imshow("Hand Detection", img)
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break;

capture.release()
cv.destroyAllWindows()