import cv2
import numpy as np


class TwoPassConnectedComponent:
    def __init__(self, img):
        self.img = img
        self.height, self.width = self.img.shape
        self.used_label = 1
        self.linked_label = {}
        self.label_occurence = {}
        self.union_find = {}

    def _find(self, label):
        if self.union_find[label] == label:
            return label
        else:
            return self._find(self.union_find[label])
    
    def _union(self, label1, label2):
        root1 = self._find(label1)
        root2 = self._find(label2)

        if root1 != root2:
            self.union_find[root2] = root1

            if root2 in self.label_occurence.keys():
                self.label_occurence[root1] += self.label_occurence[root2]
                self.label_occurence[root2] = 0
    
    def _first_pass(self):
        # Scan row by row
        for h in range(self.height):
            for w in range(self.width):
                # Ignore background
                if self.img[h, w] == 0:
                    continue

                # Get neighbors
                neighbors = []

                # Find top neighbor
                if h > 0 and self.linked_label.get((h-1, w), 0) != 0:
                    neighbors.append(self.linked_label[(h-1, w)])
                # Find top-left neighbor
                if h > 0 and w > 0 and self.linked_label.get((h-1, w-1), 0) != 0:
                    neighbors.append(self.linked_label[(h-1, w-1)])
                # Find top-right neighbor
                if h > 0 and w < self.width-1 and self.linked_label.get((h-1, w+1), 0) != 0:
                    neighbors.append(self.linked_label[(h-1, w+1)])
                # Find left neighbor
                if w > 0 and self.linked_label.get((h, w-1), 0) != 0:
                    neighbors.append(self.linked_label[(h, w-1)])
                # Find right neighbor
                if w < self.width-1 and self.linked_label.get((h, w+1), 0) != 0:
                    neighbors.append(self.linked_label[(h, w+1)])
                
                # If no neighbors, assign new label
                if len(neighbors) == 0:
                    self.linked_label[(h, w)] = self.used_label
                    self.label_occurence[self.used_label] = 1
                    self.union_find[self.used_label] = self.used_label
                    self.used_label += 1
                else:  
                    # If neighbors, assign the smallest label
                    min_label = min(neighbors)
                    self.linked_label[(h, w)] = min_label
                    self.label_occurence[min_label] += 1

                    # Perform union
                    for label in neighbors:
                        self._union(min_label, label)
    
    def _second_pass(self):
        # Iterate img through column
        for w in range(self.width):
            for h in range(self.height):
                # Ignore background
                if self.linked_label.get((h, w), 0) == 0:
                    continue
                self.linked_label[(h, w)] = self._find(self.linked_label[(h, w)])

    def run(self):
        self._first_pass()
        self._second_pass()

    def make_label_img(self):
        new_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for h in range(self.height):
            for w in range(self.width):
                if self.linked_label.get((h, w), 0) == 0:
                    continue

                # assign a hsv color to each label
                hsv = (self.linked_label[(h, w)] * 179 / self.used_label, 255, 255)

                # hsv to bgr
                bgr = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
                new_img[h, w] = bgr

        return new_img

cap = cv2.VideoCapture('./train.mp4')

if not cap.isOpened():
    print('Error opening video stream or file')

# Create background subtractor
back_subtractor = cv2.createBackgroundSubtractorMOG2()

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtractor to get mask
    foreground_mask = back_subtractor.apply(frame)
    shadow_value = back_subtractor.getShadowValue()

    # Apply threshold to mask
    ret, nmask = cv2.threshold(foreground_mask, shadow_value, 255, cv2.THRESH_BINARY)

    # Two Pass Algorithm
    two_pass = TwoPassConnectedComponent(nmask)
    two_pass.run()
    label_img = two_pass.make_label_img()
    cv2.imshow('label_img', label_img)
    print('[*] Number of labels: ', two_pass.used_label)
    print('[*] Label occurence (filtered): ', {k: v for k, v in two_pass.label_occurence.items() if v > 200})

    # Find bounding box for each label
    bounding_boxes = {}
    for h in range(two_pass.height):
        for w in range(two_pass.width):
            # Ignore background
            if two_pass.linked_label.get((h, w), 0) == 0:
                continue
            # Ignore small labels
            if two_pass.label_occurence[two_pass.linked_label[(h, w)]] < 100:
                continue

            label = two_pass.linked_label[(h, w)]
            if label not in bounding_boxes.keys():
                bounding_boxes[label] = [w, h, w, h]
            else:
                bounding_boxes[label][0] = min(bounding_boxes[label][0], w)
                bounding_boxes[label][1] = min(bounding_boxes[label][1], h)
                bounding_boxes[label][2] = max(bounding_boxes[label][2], w)
                bounding_boxes[label][3] = max(bounding_boxes[label][3], h)
    
    # Draw bounding box
    for label, box in bounding_boxes.items():
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('foreground_mask', foreground_mask)
    cv2.imshow('mask', nmask)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break