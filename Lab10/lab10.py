import cv2
import numpy as np
import time
import math
import keyboard_djitellopy
import random
from djitellopy import Tello
from pyimagesearch.pid import PID

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
    BLACK_THRESHOLD = 76800 * 0.22
    init = True
    valid_cnt = 0

    previous_pattern_1 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
    previous_pattern_2 = [1, 0, 0, 1, 1, 1, 0, 0, 0]
    previous_pattern_3 = [0, 1, 0, 0, 1, 0, 0, 1, 1]
    previous_pattern_4 = [1, 0, 0, 1, 0, 0, 1, 1, 1]
    previous_pattern_5 = [0, 1, 0, 0, 1, 0, 0, 0, 0]

    def __init__(self):
        self.fs = cv2.FileStorage("calibrate-01.xml", cv2.FILE_STORAGE_READ)
        self.intrinsic = self.fs.getNode("intrinsic").mat()
        self.distortion = self.fs.getNode("distortion").mat()

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

    def _calculate_black_area(self, cell):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        # Define lower and upper bounds for black color in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        # Threshold the HSV image to get only black colors
        mask = cv2.inRange(hsv, lower_black, upper_black)
        return np.sum(mask == 255)

    def _detect_line_in_grid(self, frame):
        height, width, _ = frame.shape
        cells = [
            frame[
                i * height // 3: (i + 1) * height // 3,
                j * width // 3: (j + 1) * width // 3,
            ]
            for i in range(3)
            for j in range(3)
        ]

        line_positions = []

        for cell in cells:
            line_position = self._calculate_black_area(
                cell
            )  # Use your line detection method here
            if line_position > self.BLACK_THRESHOLD:
                line_positions.append(1)
            else:
                line_positions.append(0)

        return line_positions

    def _draw_grid(self, frame, grid):
        height, width, _ = frame.shape

        # Draw vertical lines to divide the frame into 3 columns
        cv2.line(frame, (width // 3, 0), (width // 3, height), (255, 0, 0), 2)
        cv2.line(frame, (2 * width // 3, 0),
                 (2 * width // 3, height), (255, 0, 0), 2)

        # Draw horizontal lines to divide the frame into 3 rows
        cv2.line(frame, (0, height // 3), (width, height // 3), (255, 0, 0), 2)
        cv2.line(frame, (0, 2 * height // 3),
                 (width, 2 * height // 3), (255, 0, 0), 2)

        # draw the red rectangle on the grid if the grid is 1
        for i in range(3):
            for j in range(3):
                if grid[i * 3 + j] == 1:
                    cv2.rectangle(frame, (j * width // 3, i * height // 3),
                                  ((j + 1) * width // 3, (i + 1) * height // 3), (0, 0, 255), 2)

        return frame

    def follow_line(self, cur_direction, next_direction) -> (bool, int, int, int, int):
        """
        :param cur_direction: The current direction
        :param next_direction: The next direction
        :return: True if the feature is found and followed, False otherwise 
        """
        frame = self.frame_read.frame
        g = self._detect_line_in_grid(self.frame_read.frame)  # grids

        self._draw_grid(frame, g)
        # print("[follow_line] next_direction:", next_direction)
        print("[follow_line] grids:", g)
        # print("[follow_line] direction:", cur_direction)

        # put current direction on the frame
        cv2.putText(
            frame,
            f"cur_direction: {cur_direction}",
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        # put next direction on the frame
        cv2.putText(
            frame,
            f"next_direction: {next_direction}",
            (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("frame", frame)
        a = b = c = d = 0
        UP_SPD = 20
        DOWN_SPD = -30
        LEFT_SPD = -15
        RIGHT_SPD = 15
        BACK_SPD = -10

        # count the number of 1 in g
        count = 0
        for i in g:
            if i == 1:
                count += 1
        if count == 0:
            # No black line detected, go back
            if cur_direction == "left":
                a = RIGHT_SPD
            elif cur_direction == "up":
                c = DOWN_SPD
            elif cur_direction == "right":
                a = LEFT_SPD
            elif cur_direction == "down":
                c = UP_SPD
            # b = BACK_SPD
            return False, a, b, c, d
        if count > 5:  # Too close, go backward
            if not self.init:
                b = BACK_SPD

        if cur_direction == "left":
            a = LEFT_SPD
        elif cur_direction == "up":
            c = UP_SPD
        elif cur_direction == "right":
            a = RIGHT_SPD
        elif cur_direction == "down":
            c = DOWN_SPD

        # previous_flag = (g == self.previous_pattern_1 or g == self.previous_pattern_2 or g ==
        #                  self.previous_pattern_3 or g == self.previous_pattern_4)
        previous_flag = True
        # Check feature
        if cur_direction == "right" and next_direction == "up":  # 」↑
            if (g[1] and g[3] and g[4]) or (g[1] and g[4] and g[6] and g[7]):
                # save all kinds of patterns, for example, g[1] and g[3] and g[4] means the pattern is [0, 1, 0, 1, 1, 0, 0, 0, 0]
                self.previous_pattern_1 = [0, 1, 0, 1, 1, 0, 0, 0, 0]
                # self.previous_pattern_2 = [0, 0, 1, 1, 1, 1, 0, 0, 0]
                self.previous_pattern_3 = [0, 1, 0, 0, 1, 0, 1, 1, 0]
                # self.previous_pattern_4 = [0, 0, 1, 0, 1, 0, 1, 1, 1]
                return True, 0, 0, 0, 0
            elif (g[0] or g[1] or g[2]) and not previous_flag and not self.init:
                c = UP_SPD
            elif (g[6] or g[7] or g[8]) and not previous_flag and not self.init:
                c = DOWN_SPD

        elif cur_direction == "down" and next_direction == "left":  # 」←
            if (g[1] and g[3] and g[4]) or (g[2] and g[3] and g[4] and g[5]) or (g[1] and g[4] and g[6] and g[7]):
                self.previous_pattern_1 = [0, 1, 0, 1, 1, 0, 0, 0, 0]
                self.previous_pattern_2 = [0, 0, 1, 1, 1, 1, 0, 0, 0]
                self.previous_pattern_3 = [0, 1, 0, 0, 1, 0, 1, 1, 0]
                self.previous_pattern_4 = [0, 0, 1, 0, 1, 0, 1, 1, 1]
                return True, 0, 0, 0, 0
            elif (g[0] or g[3] or g[6]) and not previous_flag:
                a = LEFT_SPD
            elif (g[2] or g[5] or g[8]) and not previous_flag:
                a = RIGHT_SPD

        elif cur_direction == "left" and next_direction == "down":  # 「 ↓
            if (g[4] and g[5] and g[7]) or (g[1] and g[2] and g[4] and g[7]):
                self.previous_pattern_1 = [0, 0, 0, 0, 1, 1, 0, 1, 0]
                self.previous_pattern_2 = [0, 1, 1, 0, 1, 0, 0, 1, 0]
                # self.previous_pattern_3 = [0, 0, 0, 1, 1, 1, 1, 0, 0]
                self.previous_pattern_4 = [1, 1, 1, 1, 0, 0, 1, 0, 0]
                return True, 0, 0, 0, 0
            elif (g[0] or g[1] or g[2]) and not previous_flag:
                c = UP_SPD
            elif (g[6] or g[7] or g[8]) and not previous_flag:
                c = DOWN_SPD

        elif cur_direction == "up" and next_direction == "right":  # 「 →
            if (g[4] and g[5] and g[7]) or (g[3] and g[4] and g[5] and g[6]):
                self.previous_pattern_1 = [0, 0, 0, 0, 1, 1, 0, 1, 0]
                # self.previous_pattern_2 = [0, 1, 1, 0, 1, 0, 0, 1, 0]
                self.previous_pattern_3 = [0, 0, 0, 1, 1, 1, 1, 0, 0]
                self.previous_pattern_4 = [1, 1, 1, 1, 0, 0, 1, 0, 0]
                return True, 0, 0, 0, 0
            elif (g[0] or g[3] or g[6]) and not previous_flag:
                a = LEFT_SPD
            elif (g[2] or g[5] or g[8]) and not previous_flag:
                a = RIGHT_SPD
        elif cur_direction == "up" and next_direction == "left":  # 7 ←
            if (g[3] and g[4] and g[7]) or (g[3] and g[4] and g[5] and g[8]):
                self.previous_pattern_1 = [0, 0, 0, 1, 1, 0, 0, 1, 0]
                # self.previous_pattern_2 = [1, 1, 0, 0, 1, 0, 0, 1, 0]
                self.previous_pattern_3 = [0, 0, 0, 1, 1, 1, 0, 0, 1]
                self.previous_pattern_4 = [1, 1, 1, 0, 0, 1, 0, 0, 1]
                return True, 0, 0, 0, 0
            elif (g[0] or g[3] or g[6]) and not previous_flag:
                a = LEFT_SPD
            elif (g[2] or g[5] or g[8]) and not previous_flag:
                a = RIGHT_SPD

        elif cur_direction == "left" and next_direction == "up":  # L ↑
            if (g[1] and g[4] and g[5]) or (g[0] and g[3] and g[4] and g[5]) or (g[1] and g[4] and g[7] and g[8]):
                self.previous_pattern_1 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
                self.previous_pattern_2 = [1, 0, 0, 1, 1, 1, 0, 0, 0]
                self.previous_pattern_3 = [0, 1, 0, 0, 1, 0, 0, 1, 1]
                self.previous_pattern_4 = [1, 0, 0, 1, 0, 0, 1, 1, 1]
                return True, 0, 0, 0, 0
            elif (g[0] or g[1] or g[2]) and not previous_flag:
                c = UP_SPD
            elif (g[6] or g[7] or g[8]) and not previous_flag:
                c = DOWN_SPD

        return False, a, b, c, d

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
        cv2.imshow("frame", frame)

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
                        int(yaw_control),
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
                    int(z_update // 2),
                    int(vertical_update // 2),
                    int(yaw_control),
                )

        print("[follow_marker] No marker detected, stay still!")
        return False, 0, 0, 0, 0

    def follow_face(self) -> (bool, int, int, int, int):
        face_cascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")
        frame = self.frame_read.frame
        faces = face_cascade.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        two_face = []
        for face in faces:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            black_line_wid = 1 / 4 * h
            p1 = (int(x - 1 / 2 * w), int(y - black_line_wid - h / 2))
            p2 = (int(x + 3 / 2 * w), int(y - h / 2))
            p3 = (int(x - 1 / 2 * w), int(y + h / 2 + h))
            p4 = (int(x + 3 / 2 * w), int(y + h / 2 + black_line_wid + h))
            black_space = [(p1, p2), (p3, p4)]
            cv2.rectangle(frame, black_space[0][0],
                          black_space[0][1], (0, 0, 0), 2)
            cv2.rectangle(frame, black_space[1][0],
                          black_space[1][1], (0, 0, 0), 2)
            roi1 = frame[
                black_space[0][0][0]: black_space[0][1][0],
                black_space[0][0][1]: black_space[0][1][1],
            ]
            roi2 = frame[
                black_space[1][0][0]: black_space[1][1][0],
                black_space[1][0][1]: black_space[1][1][1],
            ]
            avg_c1 = roi1.mean(axis=(0, 1))
            avg_c2 = roi2.mean(axis=(0, 1))

            threshold = 120
            c1_is_black = False
            if all(avg_c1 < threshold):
                cv2.putText(
                    frame,
                    f"is Black",
                    black_space[0][0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                c1_is_black = True
            else:
                cv2.putText(
                    frame,
                    f"NOT Black!!!",
                    black_space[0][0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            c2_is_black = False
            if all(avg_c2 < threshold):
                cv2.putText(
                    frame,
                    f"is Black",
                    black_space[1][0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                c2_is_black = True
            else:
                cv2.putText(
                    frame,
                    f"NOT Black!!!",
                    black_space[1][0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            focal_length_mm = 75000
            if c1_is_black and c2_is_black:
                face_size_pixels = w * h
                distance_cm = (focal_length_mm * 15) / face_size_pixels
                two_face.append((x, y, w, h, distance_cm))
                cv2.putText(
                    frame,
                    "this pic in two face",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        if len(two_face) == 2:
            center_x = (
                (two_face[0][0] + two_face[0][2] / 2)
                + (two_face[1][0] + two_face[1][2] / 2)
            ) / 2
            center_y = (
                (two_face[0][1] + two_face[0][3] / 2)
                + (two_face[1][1] + two_face[1][3] / 2)
            ) / 2
            horizontal_offset = center_x - frame_width / 2
            vertical_offset = -1 * (center_y - frame_height / 2)
            dist = 30
            z_offset = dist - (two_face[0][4] + two_face[1][4]) / 2

            if horizontal_offset < 3 and vertical_offset < 3:
                yaw_offset = (
                    two_face[0][2] * two_face[0][3] -
                    two_face[1][2] * two_face[1][3]
                )
            else:
                yaw_offset = 0

            horizontal_update = horizontal_offset * self.SCALING_FACTOR
            vertical_update = vertical_offset * self.SCALING_FACTOR_Y
            z_update = z_offset * self.SCALING_FACTOR_Z
            yaw_update = yaw_offset * self.SCALING_FACTOR

            yaw_update = self.yaw_pid.update(yaw_update, sleep=0)
            z_update = self._clamping(self.z_pid.update(z_update, sleep=0))
            horizontal_update = self._clamping(
                self.h_pid.update(horizontal_update, sleep=0)
            )
            vertical_update = self._clamping(
                self.y_pid.update(vertical_update, sleep=0)
            )

            print(
                "[find_face]",
                "h : ",
                horizontal_update,
                "f : ",
                z_update,
                "v : ",
                vertical_update,
                "r : ",
                yaw_update,
            )
            return (
                False,
                int(horizontal_update),
                int(z_update // 2),
                int(vertical_update // 2),
                int(yaw_update),
            )

    def find_marker(self) -> list[int]:
        """
        :return: A list of marker ids found in the frame
        """
        frame = self.frame_read.frame
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            frame, self.dictionary, parameters=self.parameters
        )
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        # cv2.imshow("frame", frame)
        return markerIds

    def command_loop(
        self,
        command="stop",
        marker_id=0,
        direction=0,
        distance=0,
        next_direction=0,
    ):
        """
        This part control the drone and handle the keyboard input at the same time
        """
        # print(command, marker_id, direction, distance, next_direction)
        while True:
            frame = self.frame_read.frame
            key = cv2.waitKey(10)
            cv2.imshow("frame", frame)

            # DONT WRITE LOOPS INSIDE THIS CONDITION CHAIN
            if key != -1:
                print(key)
                keyboard_djitellopy.keyboard(self.drone, key)
            elif command == "take_off":
                self.drone.takeoff()
                return
            elif command == "land":
                self.drone.land()
                return
            elif command == "up":
                self.drone.move_up(distance)
                return
            elif command == "down":
                self.drone.move_down(distance)
                return
            elif command == "left":
                self.drone.move_left(distance)
                return
            elif command == "right":
                self.drone.move_right(distance)
                return
            elif command == "forward":
                self.drone.move_forward(distance)
                return
            elif command == "backward":
                self.drone.move_backward(distance)
                return
            elif command == "stop":
                self.send_control(0, 0, 0, 0)
                return
            elif command == "up_until_marker":
                # self.send_control(0, 0, 10, 0)
                marker_ids = self.find_marker()
                if marker_ids:
                    self.send_control(0, 0, 0, 0)
                    return
            elif command == "follow_marker":
                is_success, h, z, v, y = self.follow_marker(
                    marker_id, distance)
                if is_success:
                    print("Success follow marker!!!!!")
                    self.send_control(0, 0, 0, 0)
                    return
                self.send_control(h, z, v, y)
            elif command == "follow_face":
                is_success, h, z, v, y = self.follow_face()
                if is_success:
                    self.send_control(0, 0, 0, 0)
                    return
                self.send_control(h, z, v, y)
            elif command == "follow_line":
                is_success, h, z, v, y = self.follow_line(
                    direction, next_direction)
                if is_success:
                    self.valid_cnt += 1
                    self.send_control(0, 0, 0, 0)
                    if self.valid_cnt == 4:
                        self.valid_cnt = 0
                        return
                self.send_control(h, z, v, y)
            else:
                print("[handle_command] Invalid command:", command)
            # cv2.imshow("work", frame)


def main():
    drone = Drone()

    # 0:left, 1:up, 2:right, 3:down
    # line ['line', [0123], [line_position]] 離牆30
    actions_melody = []

    actions_carna = []

    actions_lab10 = [
        {"command": "take_off"},
        {"command": "up_until_marker"},
        {"command": "follow_marker", "marker_id": 3, "distance": 36},
        {
            "command": "follow_line",
            "direction": "right",
            "next_direction": "up",
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "right",
        },
        {
            "command": "follow_line",
            "direction": "right",
            "next_direction": "up",
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "right",
        },
        {"command": "follow_marker", "marker_id": 3, "distance": 50},
        {"command": "land"},
    ]

    active_actions = actions_lab10

    # YOLOv7 detech object to decide which action to take
    # WEIGHT = './best.pt'
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = attempt_load(WEIGHT, map_location=device)
    # if device == "cuda":
    #     model = model.half().to(device)
    # else:
    #     model = model.float().to(device)
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # while len(active_actions) == 0:
    #     image = drone.frame_read.frame
    #     image_labeled = image.copy()

    #     image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    #     if device == "cuda":
    #         image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
    #     else:
    #         image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)

    #     with torch.no_grad():
    #         output = model(image)[0]
    #     output = non_max_suppression_kpt(output, conf_thres=0.25, iou_thres=0.65)[0]

    #     output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_labeled.shape).round()
    #     for *xyxy, conf, cls in output:
    #         label = f'{names[int(cls)]} {conf:.2f}'
    #         if names[int(cls)] == 'carna':
    #             active_actions = actions_carna
    #             break
    #         elif names[int(cls)] == 'melody':
    #             active_actions = actions_melody
    #             break
    #         plot_one_box(xyxy, image_labeled, label=label, color=colors[int(cls)], line_thickness=1)

    #     cv2.waitKey(3)
    #     cv2.imshow("YOLOv7", image_labeled)
    #     cv2.destroyWindow("YOLOv7")

    for action in active_actions:
        print("[Main loop] current_action:", action)
        drone.command_loop(**action)


if __name__ == "__main__":
    main()
