# Use cv2 HOGDescriptor to detect people in the image.  Draw a rectangle around each detected person.  Use the same measure_distance function to calculate the distance to each person.  Display the distance above the rectangle.

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from math import sqrt
import imutils


# calculate the distance to an object of known width
def measure_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

imagePath = "crosswalk-featured.jpg"
image = cv2.imread(imagePath)
image = imutils.resize(image, width=min(400, image.shape[1]))

avg_body_width = 60  # Set the average width of a body in centimeters
focal_length = 600  # Set the focal length based on your camera specifications

(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), scale=1.05)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

for xA, yA, xB, yB in pick:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    # Calculate the distance from the people to the camera in meters
    distance = measure_distance(avg_body_width, focal_length, xB - xA)
    distance = distance / 100  # convert to meters
    cv2.putText(
        image,
        f"{distance:.2f} m",
        (xA, yA - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

cv2.imshow("Image", image)
cv2.waitKey(0)
