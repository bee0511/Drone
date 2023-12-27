import cv2
import numpy as np
import time
import math
import os
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
from enum import Enum


MAX_SPEED_THRESHOLD = 30
CALIBRATE_FILE = "calibrate-01.xml"

pid_controller_x = PID(kP=0.7, kI=0.0001, kD=0.1)
pid_controller_y = PID(kP=0.5, kI=0.0001, kD=0.1)
pid_controller_z = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_controller_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
height_controller_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()

# Check calibration file exists
if not os.path.isfile(CALIBRATE_FILE):
    print("Do calibrate first.")
    exit(1)

f = cv2.FileStorage(CALIBRATE_FILE, cv2.FILE_STORAGE_READ)
intrinsic = f.getNode("intrinsic").mat()
distortion = f.getNode("distortion").mat()

K_X_OFFSET = 15
K_Y_UPWARD_OFFSET = 15
K_Y_DOWNWARD_BOOST = 60


def clamping(val):
    if val > MAX_SPEED_THRESHOLD:
        return MAX_SPEED_THRESHOLD
    elif val < -MAX_SPEED_THRESHOLD:
        return -MAX_SPEED_THRESHOLD
    return int(val)


def clamping0(val):
    MAX_SPEED_THRESHOLD_0 = 40
    if val > MAX_SPEED_THRESHOLD_0:
        return MAX_SPEED_THRESHOLD_0
    elif val < -MAX_SPEED_THRESHOLD_0:
        return -MAX_SPEED_THRESHOLD_0
    return int(val)


def tracking_aruco0(frame, marker_id, z_distance):
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

    # If we see some markers
    if np.array(rvec).ndim == 0:
        return None

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
        if id != marker_id:
            continue

        rotation_matrix = np.zeros(shape=(3, 3))
        cv2.Rodrigues(rvec[i], rotation_matrix, jacobian=0)

        ypr = cv2.RQDecomp3x3(rotation_matrix)[0]
        yaw_error = ypr[1]
        x_error = tvec[0, 0, 0] - K_X_OFFSET
        if tvec[0, 0, 1] - K_Y_UPWARD_OFFSET < 0:
            y_error = -(tvec[0, 0, 1] - K_Y_UPWARD_OFFSET)
        else:
            y_error = tvec[0, 0, 1] - K_Y_DOWNWARD_BOOST
        z_error = tvec[0, 0, 2] - z_distance

        x_update = clamping0(pid_controller_x.update(x_error, sleep=0))
        y_update = clamping0(pid_controller_y.update(y_error, sleep=0))
        z_update = clamping0(pid_controller_z.update(z_error, sleep=0))
        yaw_update = clamping0(yaw_controller_pid.update(yaw_error, sleep=0))
        yaw_update = -(yaw_update + 7)
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


def tracking_aruco(frame, marker_id, z_distance):
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

    # If we see some markers
    if np.array(rvec).ndim == 0:
        return None

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
        if id != marker_id:
            continue

        rotation_matrix = np.zeros(shape=(3, 3))
        cv2.Rodrigues(rvec[i], rotation_matrix, jacobian=0)

        ypr = cv2.RQDecomp3x3(rotation_matrix)[0]
        yaw_error = ypr[1]
        x_error = tvec[0, 0, 0] - K_X_OFFSET
        if tvec[0, 0, 1] - K_Y_UPWARD_OFFSET < 0:
            y_error = -(tvec[0, 0, 1] - K_Y_UPWARD_OFFSET)
        else:
            y_error = tvec[0, 0, 1] - K_Y_DOWNWARD_BOOST
        z_error = tvec[0, 0, 2] - z_distance

        x_update = clamping(pid_controller_x.update(x_error, sleep=0))
        y_update = clamping(pid_controller_y.update(y_error, sleep=0))
        z_update = clamping(pid_controller_z.update(z_error, sleep=0))
        yaw_update = clamping(yaw_controller_pid.update(yaw_error, sleep=0))
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


class RunningStage(Enum):
    TAKEOFF_AND_FORWARD = (1,)
    FLY_OVER_BOARD = (2,)
    STOP = (100,)


def main():
    pid_controller_x.initialize()
    pid_controller_y.initialize()
    pid_controller_z.initialize()
    yaw_controller_pid.initialize()
    height_controller_pid.initialize()

    # Tello SDK
    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()

    while True:
        frame = frame_read.frame
        key = cv2.waitKey(50)
        K_TARGET_ID = 1
        packed_data = tracking_aruco(frame, K_TARGET_ID, 90)
        if not packed_data:
            if key != -1:
                keyboard(drone, key)
            else:
                drone.send_rc_control(0, 0, 0, 0)
                print("[*] Stage 1 but no arcuo 1!")
            cv2.imshow("drone", frame)
            continue

        tagged_frame, x_update, y_update, z_update, yaw_update, distance = packed_data
        if key != -1:
            keyboard(drone, key)
        elif x_update and y_update and z_update and yaw_update:  # We see marker
            # drone.send_rc_control(int(x_update), int(z_update), int(y_update), 0)
            drone.send_rc_control(
                int(x_update), int(z_update), int(y_update), int(yaw_update)
            )
            print("[*] Stage 1 Distance:", distance)
            if distance <= 100:
                break

        else:
            drone.send_rc_control(0, 0, 0, 0)

        cv2.imshow("drone", tagged_frame)

    for _ in range(20):
        frame = frame_read.frame
        print("[*] Wait for 1 second")
        drone.send_rc_control(0, 0, 0, 0)
        cv2.waitKey(50)
        cv2.imshow("drone", frame)

    # Fly over the board
    while True:
        frame = frame_read.frame
        cv2.imshow("drone", frame)
        height = drone.get_height()
        print("[*] Stage 1 fly up height:", height)
        if height < 180:  # Original 160
            print("[*] Stage 1 fly up height:", height)
            drone.send_rc_control(0, 0, 30, 0)
            cv2.waitKey(30)
        else:
            drone.send_rc_control(0, 30, 0, 0)
            cv2.waitKey(2500)
            drone.send_rc_control(0, 0, 0, 0)
            break
    # Go down
    while True:
        print("[*] Stage 2 go down")
        frame = frame_read.frame
        cv2.imshow("drone", frame)
        height = drone.get_height()
        if height > 30:
            print("[*] Stage 2 Height", height)
            drone.send_rc_control(0, 0, -40, 0)
            cv2.waitKey(30)
        else:
            drone.send_rc_control(0, 0, 0, 0)
            break

    # Track arcuo 0 & Find arcuo 3
    while True:
        frame = frame_read.frame
        key = cv2.waitKey(50)
        arcuo_3_packed = tracking_aruco(frame, 3, 50)
        if not arcuo_3_packed:
            print("[*] Find no arcuo 3!")
        else:  # Contain arcuo 3
            print("[*] Found arcuo 3!")
            (
                tagged_frame,
                x_update,
                y_update,
                z_update,
                yaw_update,
                distance,
            ) = arcuo_3_packed
            if distance <= 150:
                print("[*] Go to Stage 4!")
                break
            if key != -1:
                keyboard(drone, key)
            elif x_update and y_update and z_update and yaw_update:  # We see marker
                drone.send_rc_control(
                    int(x_update), int(z_update), int(y_update), int(yaw_update)
                )
                print("[*] Distance:", distance)
            else:
                drone.send_rc_control(0, 0, 0, 0)
            cv2.imshow("drone", tagged_frame)
            continue
        frame = frame_read.frame
        # K_TARGET_ID = 0
        packed_data = tracking_aruco0(frame, 0, 80)
        if not packed_data:
            print("[*] Find no arcuo 0!")
            if key != -1:
                keyboard(drone, key)
            else:
                drone.send_rc_control(0, 0, 0, 0)
            cv2.imshow("drone", frame)
            continue
        print("[*] Found arcuo 0!!!")
        tagged_frame, x_update, y_update, z_update, yaw_update, distance = packed_data
        if key != -1:
            keyboard(drone, key)
        elif x_update and y_update and z_update and yaw_update:  # We see marker
            drone.send_rc_control(0, int(z_update), int(y_update), int(yaw_update))
            print("[*] Distance:", distance)
            # if distance <= 60:
            #     print("[*] Too Close! Distance:", distance)
            #     break
        else:
            drone.send_rc_control(0, 0, 0, 0)
        cv2.imshow("drone", tagged_frame)

    for _ in range(20):
        drone.send_rc_control(0, 0, 0, 0)
        cv2.waitKey(50)
        print("[*] Stage 3 -> 4 wait for 1 second")
        cv2.imshow("drone", frame)
    cv2.waitKey(100)
    drone.rotate_clockwise(80)
    cv2.imshow("drone", frame)
    # Find arcuo 4
    while True:
        frame = frame_read.frame
        key = cv2.waitKey(33)
        K_TARGET_ID = 4
        packed_data = tracking_aruco(frame, K_TARGET_ID, 70)
        if not packed_data:
            if key != -1:
                keyboard(drone, key)
            else:
                print("[*] Stage 4 but no arcuo 4! Spinning...")
                drone.send_rc_control(0, 0, 0, 0)
            cv2.imshow("drone", frame)
            continue
        tagged_frame, x_update, y_update, z_update, yaw_update, distance = packed_data
        if key != -1:
            keyboard(drone, key)
        elif x_update and y_update and z_update and yaw_update:  # We see marker
            drone.send_rc_control(
                int(x_update), int(z_update), int(y_update), int(yaw_update)
            )
            print("[*] Distance:", distance)
            if distance <= 130 and int(yaw_update) <= 1:
                print("[*] Correct! Go to Stage 5!")
                break
        cv2.imshow("drone", tagged_frame)

    for _ in range(20):
        drone.send_rc_control(0, 0, 0, 0)
        cv2.waitKey(50)
        print("[*] Stage 4 -> 5 wait for 1 second")
        cv2.imshow("drone", frame)
    # Find aruco 5
    # flag_5 = False
    while True:
        frame = frame_read.frame
        key = cv2.waitKey(33)
        K_TARGET_ID = 5
        packed_data = tracking_aruco(frame, K_TARGET_ID, 130)
        if not packed_data:
            if key != -1:
                keyboard(drone, key)
            else:
                drone.send_rc_control(-20, -8, 0, 0)
                print("[*] Moving Left...")
            cv2.imshow("drone", frame)
            continue
        # if flag_5 == False:
        drone.send_rc_control(0, 0, 0, 0)
        # cv2.waitKey(1000)
        # flag_5 = True
        tagged_frame, x_update, y_update, z_update, yaw_update, distance = packed_data
        if key != -1:
            keyboard(drone, key)
        elif x_update and y_update and z_update and yaw_update:  # We see marker
            drone.send_rc_control(5, int(z_update), 0, 0)
            print("[*] Distance:", distance)
            if distance > 130 and distance < 160:
                print("[*] STOP!")
                break
        cv2.imshow("drone", tagged_frame)
    for _ in range(20):
        drone.send_rc_control(0, 0, 0, 0)
        cv2.waitKey(50)
        print("[*] Stage 5 end wait for 1 second")
        cv2.imshow("drone", frame)
    drone.land()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
