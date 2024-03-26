import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model/keras_model.h5')

# Define the labels for ASL signs
asl_signs = ['A', 'B', 'C','D','E', 'F']
# , 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Del', 'Nothing', 'Space'

# Start capturing video from the webcam
cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

# Preprocess the frame for prediction
    resized_frame = cv.resize(frame, (224,224))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 224, 224, 3))

    # Predict the sign
    prediction = model.predict(reshaped_frame)
    predicted_sign = asl_signs[np.argmax(prediction)]

    # Display the predicted sign
    cv.putText(frame, predicted_sign, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow('ASL Sign Detection', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv.destroyAllWindows()