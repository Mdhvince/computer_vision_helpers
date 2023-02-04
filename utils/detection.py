import cv2
import numpy as np
import torch


class DetectionUtils:
    def __init__(self):
        pass

    @staticmethod
    def box_area(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    @staticmethod
    def iou(box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        box_a_area = DetectionUtils.box_area(box_a)
        box_b_area = DetectionUtils.box_area(box_b)
        return inter_area / float(box_a_area + box_b_area - inter_area)

    @staticmethod
    def box_center(box):
        return int(box[0] + (box[2] - box[0]) / 2), int(box[3] + (box[1] - box[3]) / 2)

    @staticmethod
    def draw_boxes_opencv(image, boxes, thickness=1, category=None, confidence=None, color=(255, 255, 0)):
        """
        :param thickness: thickness of rectangle
        :param image: RGB or BGR Image
        :param boxes: 2D boxes (list or nd array) x1, y1, x2, y2 format
        :param category: 1D array or list of category that describe the object
        :param confidence: 1D array or list of confidence od the detection
        :param color: Tuple of 3 integers from 0-255
        :return: image
        """
        if category is None:
            category = ['']

        if confidence is None:
            confidence = ['']

        for cat, box, conf in zip(category, boxes, confidence):
            if not isinstance(box, list): box.tolist()

            x1, y1, x2, y2 = np.array([int(b) for b in box])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            if conf is not None and conf != "":
                conf = "{:.2f}".format(conf)
                cv2.putText(
                    image, f"{cat}: {conf}", (x1 - 2, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return image


if __name__ == "__main__":
    pass
