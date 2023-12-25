import numpy as np
import cv2


class OtsuThreshold:
    def __init__(self):
        self.epsilon = 1e-10  # small constant to avoid division by zero

    def otsu(self, img):
        # compute histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.max()
        Q = hist_norm.cumsum()

        bins = np.arange(256)

        fn_min = np.inf
        thresh = -1

        for i in range(1, 256):
            p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
            q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
            b1, b2 = np.hsplit(bins, [i])  # weights

            # finding means and variances
            m1, m2 = np.sum(p1 * b1) / (q1 + self.epsilon), np.sum(p2 * b2) / (
                q2 + self.epsilon
            )
            v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / (q1 + self.epsilon), np.sum(
                ((b2 - m2) ** 2) * p2
            ) / (q2 + self.epsilon)

            # calculates the minimization function
            fn = v1 * q1 + v2 * q2
            if fn < fn_min:
                fn_min = fn
                thresh = i

        return thresh

    def process_frame(self, frame):
        # compute otsu criteria
        best_threshold = self.otsu(frame)

        # Vectorized operations
        frame[frame > best_threshold] = 255
        frame[frame <= best_threshold] = 0

        return frame


if __name__ == "__main__":
    otsu_threshold = OtsuThreshold()

    for i in range(95):
        img_path = "./saved_images/saved_" + str(i) + ".png"
        img = cv2.imread(img_path, 0)

        processed_img = otsu_threshold.processFrame(img)

        cv2.imshow("My Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()