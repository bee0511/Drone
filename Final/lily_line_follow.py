# follow black line on white background

import cv2
import numpy as np
from otsu import OtsuThreshold


class LineFollow:
    def __init__(self):
        self.init = True
        self.border_thickness = 20
        self.otsu = OtsuThreshold()
        self.borders = [False, False, False, False]  # Left, Right, Top, Bottom
        self.LR_SPD = 15  # left right speed
        self.FB_SPD = 10  # forward backward speed
        self.UD_SPD = 25  # up down speed
        self.BLACK_CONST = 0.1
        self.BLACK_LINE_CONST = 0.05

    def calculate_black_area(self, cell):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        # Define lower and upper bounds for black color in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        # Threshold the HSV image to get only black colors
        mask = cv2.inRange(hsv, lower_black, upper_black)
        return np.sum(mask == 255)
    
    def detect_line(self, frame):
        height, width, _ = frame.shape
        black_threshold_leftright = width//10 * height * self.BLACK_CONST
        black_line_leftright = width//10 * height * self.BLACK_LINE_CONST
        black_threshold_topbottom = height//10 * width * self.BLACK_CONST
        black_line_topbottom = height//10 * width * self.BLACK_LINE_CONST
        # sides = the four sides of the frame
        sides = [
            frame[:height//10, :], # top
            frame[height-height//10:, :], # bottom
            frame[:, :width//10], # left
            frame[:, width-width//10:], # right
        ]
        threshold = [
            black_threshold_topbottom,
            black_threshold_topbottom,
            black_threshold_leftright,
            black_threshold_leftright,
        ]
        line_threshold = [
            black_line_topbottom,
            black_line_topbottom,
            black_line_leftright,
            black_line_leftright,
        ]

        line_position = [] # top, bottom, left, right
        for i in range(4):
            if self.calculate_black_area(sides[i]) > line_threshold[i]:
                line_position.append(2)
            elif self.calculate_black_area(sides[i]) > threshold[i]:
                line_position.append(1)
            else :
                line_position.append(0)
        
        return line_position

    def draw_sides(self, frame, sides):
        height, width, _ = frame.shape
        rect = [
            (0, 0, width, height//10),
            (0, height-height//10, width, height),
            (0, 0, width//10, height),
            (width-width//10, 0, width, height),
        ]
        blue = (255, 0, 0)
        red = (0, 0, 255)

        # draw vertical lines to divide the frame with left and right sides with width//10
        cv2.line(frame, (width//10, 0), (width//10, height), (0, 255, 0), 2)
        cv2.line(frame, (width-width//10, 0), (width-width//10, height), (0, 255, 0), 2)

        # draw horizontal lines to divide the frame with top and bottom sides with height//10
        cv2.line(frame, (0, height//10), (width, height//10), (0, 255, 0), 2)
        cv2.line(frame, (0, height-height//10), (width, height-height//10), (0, 255, 0), 2)

        # draw rectangle on the sides
        for i in range(4):
            if sides[i] == 1:
                cv2.rectangle(frame, rect[i][:2], rect[i][2:], red, 2)
            elif sides[i] == 2:
                cv2.rectangle(frame, rect[i][:2], rect[i][2:], blue, 2)

        return frame


if __name__ == "__main__":
    detector = LineFollow()
    for i in range(32, 60):
        img_path = "./saved_images/saved_" + str(i) + ".png"
        frame = cv2.imread(img_path)
        # Convert the image to grayscale
        s = detector.detect_line(frame)  # sides = [top, bottom, left, right]
        drawn_frame = detector.draw_sides(frame, s)

        # frame, is_Success, a, b, c, d = detector.detectLine(frame, "right", "up")
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
