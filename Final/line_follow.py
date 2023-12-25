# follow black line on white background

import cv2
import numpy as np
from otsu import OtsuThreshold


class LineFollow:
    def __init__(self):
        self.init = True
        self.border_thickness = 20
        self.border_length = 150
        self.otsu = OtsuThreshold()
        self.borders = [False, False, False, False]  # Left, Right, Top, Bottom
        self.LR_SPD = 15  # left right speed
        self.FB_SPD = 10  # forward backward speed
        self.UD_SPD = 25  # up down speed
        self.black_threshold = self.border_length * self.border_thickness * 0.5

    def drawBoarder(self, frame):
        height, width = frame.shape

        # Convert the frame to black and white
        bw_frame = self.otsu.process_frame(frame)

        color_frame = cv2.cvtColor(bw_frame, cv2.COLOR_GRAY2RGB)

        # Define the borders
        borders = [
            ((0, height // 2 - self.border_length),
             (self.border_thickness, height // 2 + self.border_length)),  # Left
            ((width - self.border_thickness, height // 2 - self.border_length),
             (width, height // 2 + self.border_length)),  # Right
            ((width // 2 - self.border_length, 0), (width // 2 + \
             self.border_length, self.border_thickness)),  # Top
            ((width // 2 - self.border_length, height - self.border_thickness),
             (width // 2 + self.border_length, height))  # Bottom
        ]

        positions = ['Left', 'Right', 'Top', 'Bottom']
        colors = []

        for i, ((x1, y1), (x2, y2)) in enumerate(borders):
            # Calculate the black area in the border
            black_area = np.sum(bw_frame[y1:y2, x1:x2] == 0)

            if black_area > self.black_threshold:
                # If the black area exceeds the threshold, change the border color to red and set the status to True
                color = (0, 255, 0)
                self.borders[i] = True
            else:
                # Otherwise, keep the border color green and set the status to False
                color = (0, 0, 255)
                self.borders[i] = False

            colors.append(color)

            # Draw the border
            cv2.rectangle(color_frame, (x1, y1), (x2, y2), color, 5)

        # Print the bool value at the center of the frame
        for i, (pos, val) in enumerate(zip(positions, self.borders)):
            position = (50, 50 + i * 30)
            text = f'{pos}: {str(val)}'
            cv2.putText(color_frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 1, cv2.LINE_AA)

        return color_frame


if __name__ == "__main__":
    detector = LineFollow()
    for i in range(32, 60):
        img_path = "./saved_images/saved_" + str(i) + ".png"
        frame = cv2.imread(img_path)
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = detector.drawBoarder(gray_img)
        print(detector.borders)
        # frame, is_Success, a, b, c, d = detector.detectLine(frame, "right", "up")
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
