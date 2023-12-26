import cv2
import numpy as np
import time
import math
import keyboard_djitellopy
import random
from djitellopy import Tello
from pyimagesearch.pid import PID
from face_detect import FaceDetector
from line_follow import LineFollow

# YOLOv7
import torch
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import plot_one_box


def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


class Drone:
    MAX_SPEED_THRESHOLD = 30
    SCALING_FACTOR = 1
    SCALING_FACTOR_H = 0.2
    SCALING_FACTOR_Y = 0.4
    SCALING_FACTOR_Z = 0.3
    HAND_CONTROL_FB = 30    # Front and back
    HAND_CONTROL_LR = 20    # Left and right
    HAND_CONTROL_UD = 35    # Up and down
    init = True
    time_cnt = 0
    wait_key_interval = 10
    interrupted_flag = False

    def __init__(self):
        self.fs = cv2.FileStorage("calibrate-01.xml", cv2.FILE_STORAGE_READ)
        self.intrinsic = self.fs.getNode("intrinsic").mat()
        self.distortion = self.fs.getNode("distortion").mat()

        # Face detector
        self.face_detector = FaceDetector()

        # Line follower
        self.line_follower = LineFollow()

        # Aruco marker
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Tello and frame_read object
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()
        self.frame_read = self.drone.get_frame_read()

        # PID Controller
        self.z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.h_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.yaw_pid.initialize()
        self.z_pid.initialize()
        self.h_pid.initialize()
        self.y_pid.initialize()

    def send_control(
        self,
        left_right_velocity,
        forward_backward_velocity,
        up_down_velocity,
        yaw_velocity,
    ):
        self.drone.send_rc_control(
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity,
        )

    def _clamping(self, val):
        if val > self.MAX_SPEED_THRESHOLD:
            return self.MAX_SPEED_THRESHOLD
        elif val < -self.MAX_SPEED_THRESHOLD:
            return -self.MAX_SPEED_THRESHOLD
        return int(val)

    def find_marker(self) -> list[int]:
        """
        :return: A list of marker ids found in the frame
        """
        frame = self.frame_read.frame
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            frame, self.dictionary, parameters=self.parameters
        )
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        return frame, markerIds

    def follow_marker(self, marker_id, distance) -> (bool, int, int, int, int):
        """
        :param marker_id: The id of the marker to follow
        :param distance: The distance to keep from the marker
        :return: True if the marker if found and reached, False otherwise
        """
        frame = self.frame_read.frame
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            frame, self.dictionary, parameters=self.parameters
        )
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            markerCorners, 15, self.intrinsic, self.distortion
        )
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        cv2.imshow("marker", frame)

        if np.array(rvec).ndim == 0:
            return False, 0, 0, 0, 0

        frame_width, frame_height = frame.shape[1], frame.shape[0]

        for i in range(len(markerIds)):
            if markerIds[i] == marker_id:
                center_x = (markerCorners[i][0][0]
                            [0] + markerCorners[i][0][2][0]) / 2
                center_y = (markerCorners[i][0][0]
                            [1] + markerCorners[i][0][2][1]) / 2
                horizontal_offset = center_x - frame_width / 2
                vertical_offset = -1 * (center_y - frame_height / 2)

                horizontal_update = horizontal_offset * self.SCALING_FACTOR_H
                vertical_update = (vertical_offset - 50) * \
                    self.SCALING_FACTOR_Y

                rot_mat, _ = cv2.Rodrigues(rvec[i])
                euler_angles = cv2.RQDecomp3x3(rot_mat)
                yaw_angle = euler_angles[0][2]
                yaw_control = yaw_angle * self.SCALING_FACTOR

                z_update = tvec[i, 0, 2] - distance
                if (
                    z_update < 15
                    and z_update > -15
                    and yaw_angle < 15
                    and yaw_angle > -15
                ):
                    return (
                        True,
                        int(horizontal_update),
                        int(z_update // 2),
                        int(vertical_update // 2),
                        self._clamping(int(yaw_control)),
                    )

                z_update = self._clamping(self.z_pid.update(z_update, sleep=0))
                horizontal_update = self._clamping(
                    self.h_pid.update(horizontal_update, sleep=0)
                )
                vertical_update = self._clamping(
                    self.y_pid.update(vertical_update, sleep=0)
                )
                print(
                    "[follow_marker] Target marker detected",
                    "h : ",
                    horizontal_update,
                    "f : ",
                    z_update,
                    "v : ",
                    vertical_update,
                    "r : ",
                    yaw_control,
                )
                return (
                    False,
                    int(horizontal_update),
                    int(z_update),
                    int(vertical_update // 2),
                    int(yaw_control),
                )

        print("[follow_marker] No marker detected, stay still!")
        return False, 0, 0, 0, 0

    def follow_line(self, cur_direction, next_direction) -> (np.ndarray, bool, int, int, int, int):
        frame = self.frame_read.frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = self.line_follower.drawBoarder(gray_frame)
        detect_array = self.line_follower.borders  # Left, Right, Top, Bottom

        # put current direction and next direction on the frame
        cv2.putText(frame, f"cur: {cur_direction}", (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"next: {next_direction}", (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # if not any(detect_array):
        #     return frame, False, 0, 0, 0, 0

        if self.init == True:
            if cur_direction == "right":
                return frame, False, self.line_follower.LR_SPD, -self.line_follower.FB_SPD, 0, 0
            if cur_direction == "left":
                return frame, False, -self.line_follower.LR_SPD, -self.line_follower.FB_SPD, 0, 0
            if cur_direction == "up":
                return frame, False, 0, -self.line_follower.FB_SPD, self.line_follower.UD_SPD, 0
            if cur_direction == "down":
                return frame, False, 0, -self.line_follower.FB_SPD, -self.line_follower.UD_SPD, 0
            if cur_direction == "left_table":
                return frame, False, -self.line_follower.LR_SPD, 0, 0, 0

        if cur_direction == "right" and next_direction == "down":
            if detect_array[3]:
                return frame, True, 0, 0, 0, 0
            return frame, False, self.line_follower.LR_SPD, 0, 0, 0
        elif cur_direction == "right" and next_direction == "up":
            if detect_array[2]:
                return frame, True, 0, 0, 0, 0
            return frame, False, self.line_follower.LR_SPD, 0, 0, 0
        elif cur_direction == "up" and next_direction == "right":
            if detect_array[1]:
                return frame, True, 0, 0, 0, 0
            return frame, False, 0, 0, self.line_follower.UD_SPD, 0
        elif cur_direction == "up" and next_direction == "left":
            if detect_array[0]:
                return frame, True, 0, 0, 0, 0
            return frame, False, 0, 0, self.line_follower.UD_SPD, 0
        elif cur_direction == "left" and next_direction == "down":
            if detect_array[3]:
                return frame, True, 0, 0, 0, 0
            return frame, False, -self.line_follower.LR_SPD, 0, 0, 0
        elif cur_direction == "left" and next_direction == "up":
            if detect_array[2]:
                return frame, True, 0, 0, 0, 0
            return frame, False, -self.line_follower.LR_SPD, 0, 0, 0
        elif cur_direction == "down" and next_direction == "right":
            if detect_array[1]:
                return frame, True, 0, 0, 0, 0
            return frame, False, 0, 0, -self.line_follower.UD_SPD, 0
        elif cur_direction == "down" and next_direction == "left":
            if detect_array[0]:
                return frame, True, 0, 0, 0, 0
            return frame, False, 0, 0, -self.line_follower.UD_SPD, 0
        elif cur_direction == "left" and next_direction == "aruco":
            frame, markerIds = self.find_marker()
            if markerIds is not None:
                return frame, True, 0, 0, 0, 0
            return frame, False, -self.line_follower.LR_SPD, -10, 0, 0
        elif cur_direction == "left_table" and next_direction == "up_table":
            if detect_array[2]:
                return frame, True, 0, 0, 0, 0
            return frame, False, -self.line_follower.LR_SPD, 0, 0, 0
        else:
            print("[follow_line] Invalid direction:",
                  cur_direction, next_direction)

    def follow_face(self) -> (bool, int, int, int, int):
        frame = self.frame_read.frame
        frame, dist = self.face_detector.detect(frame)
        if dist is None:
            cv2.putText(frame, "No face detected", (150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame, False, 10, -5, 0, 0
        if dist > self.face_detector.error:
            return frame, False, -7, 0, 0, 0
        if dist < -self.face_detector.error:
            return frame, False, 7, 0, 0, 0
        return frame, True, 0, 0, 0, 0

    def command_loop(
        self,
        command="stop",
        marker_id=0,
        direction=0,
        distance=0,
        next_direction=0,
        valid_count=1,
        height=0,
        time_threshold=100,
    ):
        """
        This part control the drone and handle the keyboard input at the same time
        """
        tmp_valid_count = valid_count
        while True:
            frame = self.frame_read.frame
            # put command on screen
            cv2.putText(frame, f"command: {command}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            key = cv2.waitKey(self.wait_key_interval)

            # DONT WRITE LOOPS INSIDE THIS CONDITION CHAIN
            if key != -1 and key != ord("p"):
                print(key)
                keyboard_djitellopy.keyboard(self.drone, key)
            elif command == "take_off":
                self.drone.takeoff()
                return
            elif command == "land":
                self.drone.land()
                return
            elif command == "turn_right":
                self.drone.rotate_clockwise(90)
                return
            elif command == "turn_back":
                self.drone.rotate_counter_clockwise(180)
                return
            elif command == "stop":
                self.send_control(0, 0, 0, 0)
                return
            elif command == "forward":
                valid_count -= 1
                cv2.putText(frame, f"valid_count: {valid_count}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if valid_count == 0:
                    self.send_control(0, 0, 0, 0)
                    return
                self.send_control(0, self.HAND_CONTROL_FB, 0, 0)
            elif command == "up_until_marker":
                self.send_control(0, 0, 0, 0)
                frame, marker_ids = self.find_marker()
                print(marker_ids)
                if marker_ids is not None:
                    self.send_control(0, 0, 0, 0)
                    return
            elif command == "follow_marker":
                is_success, x, z, y, yaw = self.follow_marker(
                    marker_id, distance)
                if is_success or key == ord("p"):
                    print("Success follow marker!!!!!")
                    self.send_control(0, 0, 0, 0)
                    return
                self.send_control(x, z, y, yaw)
            elif command == "follow_face":
                frame, is_success, x, z, y, yaw = self.follow_face()
                # Put x, z, y, yaw on the frame
                if is_success:
                    valid_count -= 1
                    self.send_control(0, 0, 0, 0)
                    cv2.putText(frame, f"valid_count: {valid_count}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if valid_count <= 0:
                        return
                else:
                    valid_count = tmp_valid_count
                cv2.putText(frame, f"x: {x}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"z: {z}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"y: {y}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"yaw: {yaw}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.send_control(x, z, y, yaw)
            elif command == "follow_line":
                if self.init == True:
                    self.time_cnt += self.wait_key_interval
                    if self.time_cnt > time_threshold:
                        self.init = False
                        self.time_cnt = 0
                frame, is_success, x, z, y, yaw = self.follow_line(
                    direction, next_direction)
                cv2.putText(frame, f"timer: {self.time_cnt}", (150, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if is_success:
                    valid_count -= 1
                    self.send_control(0, 0, 0, 0)
                    cv2.putText(frame, f"valid_count: {valid_count}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if valid_count == 0:
                        self.init = True
                        return
                else:
                    valid_count = tmp_valid_count
                # Put x, z, y, yaw on the frame
                cv2.putText(frame, f"x: {x}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"z: {z}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"y: {y}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"yaw: {yaw}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.send_control(x, z, y, yaw)
            else:
                print("[handle_command] Invalid command:", command)

            cv2.imshow("frame", frame)


def main():
    drone = Drone()

    actions_test = [
        {"command": "up_until_marker", "marker_id": 2},
        {"command": "follow_marker", "marker_id": 2, "distance": 305},
        {"command": "turn_right"},
        # {"command": "follow_face", "valid_count": 1},
        {"command": "forward", "valid_count": 500},
        {"command": "turn_back"},
        {"command": "follow_marker", "marker_id": 3, "distance": 150},
        {"command": "land"},
    ]

    actions_melody = [
        {"command": "forward", "valid_count": 600},
        {"command": "up_until_marker"},
        {"command": "follow_marker", "marker_id": 1, "distance": 36},
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "up",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left_table",
            "next_direction": "up_table",
            "valid_count": 1,
            "time_threshold": 4000,
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "aruco",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {"command": "follow_marker", "marker_id": 2, "distance": 305},
        {"command": "turn_right"},
        # {"command": "follow_face", "valid_count": 5},
        {"command": "forward", "valid_count": 400},
        {"command": "turn_back"},
        {"command": "follow_marker", "marker_id": 3, "distance": 155},
        {"command": "land"},
    ]

    actions_carna = [
        {"command": "forward", "valid_count": 600},
        {"command": "up_until_marker"},
        {"command": "follow_marker", "marker_id": 1, "distance": 36},
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left_table",
            "next_direction": "up_table",
            "valid_count": 1,
            "time_threshold": 4000,
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "up",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "left",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "aruco",
            "valid_count": 1,
            "time_threshold": 100,
        },
        {"command": "follow_marker", "marker_id": 2, "distance": 305},
        {"command": "turn_right"},
        # {"command": "follow_face", "valid_count": 5},
        {"command": "forward", "valid_count": 400},
        {"command": "turn_back"},
        {"command": "follow_marker", "marker_id": 3, "distance": 155},
        {"command": "land"},
    ]

    action_all = [
        {"command": "up_until_marker"},
        {"command": "follow_marker", "marker_id": 1, "distance": 36},
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "up",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "left",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "left",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "left_table",
            "next_direction": "up_table",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "up",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "left",
            "valid_count": 1,
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "aruco",
            "valid_count": 1,
        },
        {"command": "follow_marker", "marker_id": 2, "distance": 305},
        {"command": "turn_right"},
        # {"command": "follow_face", "valid_count": 5},
        {"command": "forward", "valid_count": 400},
        {"command": "turn_back"},
        {"command": "follow_marker", "marker_id": 3, "distance": 155},
        {"command": "land"},
    ]

    # active_actions = action_all
    # active_actions = actions_test

    drone.drone.takeoff()
    while True:
        frame = drone.frame_read.frame
        key = cv2.waitKey(10)
        cv2.imshow("frame", frame)
        if key == ord("n"):
            active_actions = actions_carna
            print("action set to carna")
            break
        elif key == ord("m"):
            active_actions = actions_melody
            print("action set to melody")
            break
        else:
            print("Invalid key!")

    for action in active_actions:
        print("[Main loop] current_action:", action)
        drone.command_loop(**action)


if __name__ == "__main__":
    main()