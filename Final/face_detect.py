# a class that can detect faces in an image and calculate the distance to the faces
import cv2


class FaceDetector:
    def __init__(self):
        self.XMLFILE = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(self.XMLFILE)
        self.focal_length = 650  # Set the focal length based on your camera specifications
        self.avg_face_width = 14  # Set the average width of a face in centimeters
        self.midpoint_offset = 5

    def measureFaceDistance(self, face_width_pixels):
        return (self.avg_face_width * self.focal_length) / face_width_pixels

    def drawTwoFaceCenter(self, frame, centers):
        # If there are exactly two faces, draw a 10x10 square at the midpoint of the two centers
        if len(centers) == 2:
            midpoint_x = sum(x for x, y in centers) // 2
            midpoint_y = sum(y for x, y in centers) // 2
            cv2.rectangle(frame, (midpoint_x - self.midpoint_offset, midpoint_y - self.midpoint_offset),
                          (midpoint_x + self.midpoint_offset, midpoint_y + self.midpoint_offset), (0, 0, 255), -1)
        else:
            midpoint_x = None
            midpoint_y = None
        return frame

    def drawImgCenter(self, frame):
        img_height, img_width, _ = frame.shape
        midpoint_x = img_width // 2
        midpoint_y = img_height // 2
        cv2.rectangle(frame, (midpoint_x - self.midpoint_offset, midpoint_y - self.midpoint_offset),
                      (midpoint_x + self.midpoint_offset, midpoint_y + self.midpoint_offset), (0, 255, 0), -1)
        return frame

    def calculateDistance(self, frame, centers):
        # calculate distance between the center of two faces and the center of the image
        img_height, img_width, _ = frame.shape
        midpoint_x = img_width // 2
        midpoint_y = img_height // 2
        if len(centers) == 2:
            face_midpoint_x = sum(x for x, y in centers) // 2
            distance = midpoint_x - face_midpoint_x
            cv2.putText(
                frame,
                f"{distance:.2f} pixels",
                (midpoint_x, midpoint_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        else:
            distance = None
        # put the distance on the frame
        return frame, distance

    def detectFaces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # 儲存所有偵測到的人臉的中心點
        centers = []

        for x, y, w, h in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 255), 2)

            # Calculate the width of the detected face in pixels
            face_width_pixels = w

            # Measure the distance to the detected face
            distance = self.measureFaceDistance(face_width_pixels)

            # Display the distance above the face rectangle
            cv2.putText(
                frame,
                f"{distance:.2f} cm",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 100, 255),
                2,
            )

            # Calculate the center of the face and add it to the list
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        return frame, centers

    def detect(self, frame):
        """Detect faces in the frame and return the frame with the distance to the faces

        Args:
            frame (np.array): the frame to detect faces

        Returns:
            np.array: the modified frame
            float: the distance between the center of the frame and the center of the faces
        """
        frame, centers = detector.detectFaces(frame)
        frame = detector.drawTwoFaceCenter(frame, centers)
        frame = detector.drawImgCenter(frame)
        frame, dist = detector.calculateDistance(frame, centers)

        return frame, dist


if __name__ == "__main__":
    detector = FaceDetector()
    for i in range(32, 60):
        img_path = "./saved_images/saved_" + str(i) + ".png"
        frame = cv2.imread(img_path)
        # frame, centers = detector.detectFaces(frame)
        # frame = detector.drawTwoFaceCenter(frame, centers)
        # frame = detector.drawImgCenter(frame)
        # frame, dist = detector.calculateDistance(frame, centers)
        frame, dist = detector.detect(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
