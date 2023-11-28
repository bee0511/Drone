import cv2
import numpy as np
import time
import math
import os
from cali import Calibration
from djitellopy import Tello
from pyimagesearch.pid import PID

CALIBRATE_FILE = "calibrate-01.xml"

def main():
    # Check calibration file exists
    if not os.path.isfile(CALIBRATE_FILE):
        print("Do calibrate first.")
        exit(1)

    f = cv2.FileStorage(CALIBRATE_FILE, cv2.FILE_STORAGE_READ)
    intrinsic = f.getNode("intrinsic").mat()
    distortion = f.getNode("distortion").mat()

    print(intrinsic)
    print(distortion)

    # Tello SDK
    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()

    # Load the predefined dictionary
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    while True:
        frame = frame_read.frame

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
            frame, dictionary, parameters=parameters
        )
        modified_frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            markerCorners, 15, intrinsic, distortion
        )
        if np.array(rvec).ndim == 0:
            cv2.imshow("drone", modified_frame)
            key = cv2.waitKey(33)
            continue

        for i in range(rvec.shape[0]):
            modified_frame = cv2.aruco.drawAxis(
                modified_frame, intrinsic, distortion, rvec[i, :, :], tvec[i, :, :], 10
            )
            cv2.putText(
                frame,
                f"x = {tvec[0,0,0]}, y = {tvec[0,0,1]}, z = {tvec[0,0,2]}",
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("drone", modified_frame)
        key = cv2.waitKey(33)

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
