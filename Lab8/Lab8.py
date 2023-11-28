import cv2
import numpy as np
from math import sqrt


def measure_distance(face_width_pixels):
    return (avg_face_width * focal_length) / face_width_pixels


def dis(tvec):
    return round(float(tvec[2]), 2)


XMLFILE = "haarcascade_frontalface_default.xml"


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

face_cascade = cv2.CascadeClassifier(XMLFILE)
cap = cv2.VideoCapture(0)

f = cv2.FileStorage("calibrate.xml", cv2.FILE_STORAGE_READ)
intrinsic = f.getNode("intrinsic").mat()
distortion = f.getNode("distortion").mat()

focal_length = 600  # Set the focal length based on your camera specifications
avg_face_width = 14  # Set the average width of a face in centimeters

body_y = 200
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
        distance = measure_distance(face_width_pixels)

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

    rects, weights = hog.detectMultiScale(
        gray, winStride=(8, 8), scale=1.05, useMeanshiftGrouping=False
    )
    for rect in rects:
        (x, y, w, h) = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 255), 2)
        imgPoints = np.float32([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        body_x = body_y / 2
        print(body_x, body_y)
        body_objPoints = np.float32(
            [(0, 0, 0), (body_x, 0, 0), (body_x, body_y, 0), (0, body_y, 0)]
        )
        _, _, tvec = cv2.solvePnP(body_objPoints, imgPoints, intrinsic, distortion)
        # print(tvec)
        cv2.putText(
            frame,
            str(dis(tvec)),
            (x + w, y + h),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 100, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("window", frame)
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
