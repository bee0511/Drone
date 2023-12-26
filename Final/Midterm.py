import cv2
import numpy as np
import time
import math
import os
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
from enum import Enum

class track_aruco:
    
    def __init__(self):
        self.MAX_SPEED_THRESHOLD = 30
        self.CALIBRATE_FILE = "calibrate-01.xml"

        self.pid_controller_x = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.pid_controller_y = PID(kP=0.5, kI=0.0001, kD=0.1)
        self.pid_controller_z = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.yaw_controller_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.height_controller_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        # Load the predefined dictionary
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Initialize the detector parameters using default values
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.f = cv2.FileStorage(self.CALIBRATE_FILE, cv2.FILE_STORAGE_READ)
        self.intrinsic = self.f.getNode("intrinsic").mat()
        self.distortion = self.f.getNode("distortion").mat()
        # self.K_X_OFFSET = 15
        self.K_Y_UPWARD_OFFSET = 15
        self.K_Y_DOWNWARD_BOOST = 60
        
        self.pid_controller_x.initialize()
        self.pid_controller_y.initialize()
        self.pid_controller_z.initialize()
        self.yaw_controller_pid.initialize()
        self.height_controller_pid.initialize()


    def clamping(self, val):
        if val > self.MAX_SPEED_THRESHOLD:
            return self.MAX_SPEED_THRESHOLD
        elif val < -self.MAX_SPEED_THRESHOLD:
            return -self.MAX_SPEED_THRESHOLD
        return int(val)


    def tracking_aruco(self, frame, marker_id, z_distance):
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
            frame, self.dictionary, parameters=self.parameters
        )
        modified_frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            markerCorners, 15, self.intrinsic, self.distortion
        )
        z_update = None
        y_update = None
        yaw_update = None

        # If we see some markers
        if np.array(rvec).ndim == 0:
            return None

        # Iterate all markers
        for i in range(rvec.shape[0]):
            modified_frame = cv2.aruco.drawAxis(
                modified_frame,
                self.intrinsic,
                self.distortion,
                rvec[i, :, :],
                tvec[i, :, :],
                10,
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

            id = markerIds[i][0]
            if id != marker_id:
                continue

            rotation_matrix = np.zeros(shape=(3, 3))
            cv2.Rodrigues(rvec[i], rotation_matrix, jacobian=0)

            ypr = cv2.RQDecomp3x3(rotation_matrix)[0]
            yaw_error = ypr[1]
            x_error = tvec[0, 0, 0]
            if tvec[0, 0, 1] - self.K_Y_UPWARD_OFFSET < 0:
                y_error = -(tvec[0, 0, 1] - self.K_Y_UPWARD_OFFSET)
            else:
                y_error = tvec[0, 0, 1] - self.K_Y_DOWNWARD_BOOST
            z_error = tvec[0, 0, 2] - z_distance

            x_update = self.clamping(self.pid_controller_x.update(x_error, sleep=0))
            y_update = self.clamping(self.pid_controller_y.update(y_error, sleep=0))
            z_update = self.clamping(self.pid_controller_z.update(z_error, sleep=0))
            yaw_update = self.clamping(self.yaw_controller_pid.update(yaw_error, sleep=0))
            # yaw_update = -(yaw_update + 7)
            # print("[DEBUG][NOTCLAMP] ", ypr)
            # print(
            #     "[DEBUG][NOTCLAMP] ",
            #     "x=",
            #     x_error,
            #     "y=",
            #     y_error,
            #     "z=",
            #     z_error,
            #     "yaw=",
            #     yaw_error,
            # )
            # print("[DEBUG][CLAMPPED] ", x_update, y_update, z_update, yaw_update)
            return (modified_frame, x_update, y_update, z_update, yaw_update, tvec[0, 0, 2])

        return None


def main():
    pass
if __name__ == "__main__":
    main()
