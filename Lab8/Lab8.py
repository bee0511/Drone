import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from math import sqrt


def measureFaceDistance(face_width_pixels):
    return (avg_face_width * focal_length) / face_width_pixels


def measureBodyDistance(body_width_pixels):
    return (avg_body_width * focal_length) / body_width_pixels


XMLFILE = "haarcascade_frontalface_default.xml"


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

face_cascade = cv2.CascadeClassifier(XMLFILE)
cap = cv2.VideoCapture(0)

focal_length = 650  # Set the focal length based on your camera specifications
avg_face_width = 14  # Set the average width of a face in centimeters
avg_body_width = 77.5  # Set the average width of a body in centimeters

while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the width of the detected face in pixels
        face_width_pixels = w

        # Measure the distance to the detected face
        distance = measureFaceDistance(face_width_pixels)

        # Display the distance above the face rectangle
        cv2.putText(
            frame,
            f"{distance:.2f} cm",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    rects, weights = hog.detectMultiScale(gray, winStride=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for xA, yA, xB, yB in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 100, 255), 2)
        # Calculate the distance from the people to the camera in meters
        body_width_pixels = xB - xA
        distance = measureBodyDistance(body_width_pixels) / 100  # convert to meters
        cv2.putText(
            frame,
            f"{distance:.2f} m",
            (xA, yA - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 100, 255),
            2,
        )
    cv2.imshow("window", frame)
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
