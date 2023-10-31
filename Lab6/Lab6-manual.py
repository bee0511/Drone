import cv2
import numpy as np
import time
import math
import os
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard


MAX_SPEED_THRESHOLD = 45


def clamping(val):
    if val > MAX_SPEED_THRESHOLD:
        return MAX_SPEED_THRESHOLD
    elif val < -MAX_SPEED_THRESHOLD:
        return -MAX_SPEED_THRESHOLD
    return int(val)


def main():
    # Check calibration file exists
    if not os.path.isfile("calibrate.xml"):
        print("Do calibrate first.")
        exit(1)

    f = cv2.FileStorage("calibrate-best.xml", cv2.FILE_STORAGE_READ)
    intrinsic = f.getNode("intrinsic").mat()
    distortion = f.getNode("distortion").mat()
    pid_controller_x = PID(kP=0.7, kI=0.0001, kD=0.1)
    pid_controller_y = PID(kP=0.7, kI=0.0001, kD=0.1)
    pid_controller_z = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_controller_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    pid_controller_x.initialize()
    pid_controller_y.initialize()
    pid_controller_z.initialize()
    yaw_controller_pid.initialize()

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
        z_update = None
        y_update = None
        yaw_update = None

        key = cv2.waitKey(33)
        # If we see some markers
        if np.array(rvec).ndim != 0:
            # Iterate all markers
            for i in range(rvec.shape[0]):
                modified_frame = cv2.aruco.drawAxis(
                    modified_frame,
                    intrinsic,
                    distortion,
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
                if id != 0:
                    continue

                rotation_matrix = np.zeros(shape=(3, 3))
                cv2.Rodrigues(rvec[i], rotation_matrix, jacobian=0)
                # res = cv2.Rodrigues(rvec[0])[0]
                # res = res @ np.array([0, 0, 1])

                ypr = cv2.RQDecomp3x3(rotation_matrix)[0]
                yaw_error = ypr[1]
                x_error = tvec[0, 0, 0] - 40
                if tvec[0, 0, 1] - 15 < 0:
                    y_error = -(tvec[0, 0, 1] - 15)
                else:
                    y_error = tvec[0, 0, 1] - 60
                z_error = tvec[0, 0, 2] - 80

                x_update = clamping(pid_controller_x.update(x_error, sleep=0))
                y_update = clamping(pid_controller_y.update(y_error, sleep=0))
                z_update = clamping(pid_controller_z.update(z_error, sleep=0))
                # yaw_update = clamping(yaw_controller_pid.update(yaw_error, sleep=0))
                # yaw_error = math.atan2(res[0], res[2]) * 5
                yaw_update = clamping(yaw_controller_pid.update(yaw_error, sleep=0))
                yaw_update = -(yaw_update + 7)
                print("[DEBUG][NOTCLAMP] ", ypr)
                print(
                    "[DEBUG][NOTCLAMP] ",
                    "x=",
                    x_error,
                    "y=",
                    y_error,
                    "z=",
                    z_error,
                    "yaw=",
                    yaw_error,
                )
                print("[DEBUG][CLAMPPED] ", x_update, y_update, z_update, yaw_update)

                if key != -1:
                    keyboard(drone, key)
                elif x_error and y_update and z_update and yaw_update:
                    drone.send_rc_control(
                        0, int(z_update), int(y_update), int(yaw_update)
                    )
        else:
            if key != -1:
                keyboard(drone, key)
            else:
                drone.send_rc_control(0, 0, 0, 0)

        cv2.imshow("drone", modified_frame)

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
