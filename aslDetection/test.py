import cv2 as cv
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from termcolor import colored


capture = cv.VideoCapture(1)
detector = HandDetector(maxHands = 1)
classifier = Classifier(modelPath = 'model/keras_model.h5', labelsPath = 'model/labels.txt')

offset = 20
img_size = 400

folder = 'data/A' # Replace A or B or C to save to required folder.
counter = 0
labels = ["A", "B", "C","D","E","F"]
# "G","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","H"
while True:
    isTrue, frame = capture.read()
    output_img = frame.copy()
    hands, img = detector.findHands(frame, draw = False)
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
                predct, index = classifier.getPrediction(white_img)
                # print(predct, index)
            else:
                k = img_size / w
                nheight = math.ceil(k * h)
                resized_img = cv.resize(cropped_img, (img_size, nheight))
                resized_img_shape = resized_img.shape
                # print(colored(f"Resized - Height: {resized_img_shape[0]}  Width: {resized_img_shape[1]}", "blue"))
                height_gap = math.ceil((img_size - nheight) / 2)
                # white_img[height_gap:resized_img_shape[0] + height_gap, :] = resized_img
                white_img[height_gap:resized_img_shape[0] + height_gap, :] = resized_img
                predct, index = classifier.getPrediction(white_img)
                # print(predct, index)

            cv.rectangle( output_img , ( x - offset , y - offset - 50 ) , ( x - offset + 90 , y - offset - 50 + 50 ) , ( 255 , 0 , 255 ) , cv.FILLED )
            cv.putText(output_img, labels[index], (x, y - 26), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2) 
            cv.rectangle(output_img,  (x - offset, y - offset) , (x + w + offset , y + h + offset) , ( 255 , 0 , 255 ) , 4 )

            # cv.imshow("Hand Image", cropped_img)
            cv.imshow("White Hand Image", white_img)


    # cv.imshow("Hand Detection", img)
    cv.imshow("Hand Detection", output_img)

    # key = cv.waitKey(1)
    # if key == ord('c'):
    #     cv.imwrite(f'{folder}/Image_{time.time()}.jpg', white_img)
    #     counter += 1
    #     print(colored(counter, "grey"))
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break;

capture.release()
cv.destroyAllWindows()