import cv2 as cv
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from termcolor import colored


capture = cv.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20
img_size = 400

folder = 'Data/C' # Replace A or B or C to save to required folder.
counter = 0

while True:
    isTrue, frame = capture.read()
    hands, img = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y , w, h = hand['bbox']
        white_img = np.ones((img_size, img_size, 3), np.uint8) * 255

        if (y - offset >= 0) and (y + h + offset < img.shape[0]) and (x - offset >= 0) and (x + w + offset < img.shape[0]):

            cropped_img = img[y - offset: y + h + offset, x - offset: x + w + offset]

            cropped_img_shape = cropped_img.shape
            # print(colored(f"Cropped Height: {cropped_img_shape[0]}  Width: {cropped_img_shape[1]}", "green"))

            aspect_ratio = h / w
            if aspect_ratio > 1:
                k = img_size / h
                nwidth = math.ceil(k * w)
                resized_img = cv.resize(cropped_img, (nwidth, img_size))
                resized_img_shape = resized_img.shape
                # print(colored(f"Resized - Height: {resized_img_shape[0]}  Width: {resized_img_shape[1]}", "blue"))
                width_gap = math.ceil((img_size - nwidth) / 2)
                white_img[:, width_gap:resized_img_shape[1] + width_gap] = resized_img
            else:
                k = img_size / w
                nheight = math.ceil(k * h)
                resized_img = cv.resize(cropped_img, (img_size, nheight))
                resized_img_shape = resized_img.shape
                # print(colored(f"Resized - Height: {resized_img_shape[0]}  Width: {resized_img_shape[1]}", "blue"))
                height_gap = math.ceil((img_size - nheight) / 2)
                white_img[height_gap:resized_img_shape[0] + height_gap, :] = resized_img

            cv.imshow("Hand Image", cropped_img)
            cv.imshow("White Hand Image", white_img)
    
    cv.imshow("Hand Detection", img)

    key = cv.waitKey(1)
    if key == ord('c'):
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', white_img)
        counter += 1
        print(colored(counter, "grey"))
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break;

capture.release()
cv.destroyAllWindows()