import cv2


def draw_boxes_opencv(image, boxes, category=None, confidence=None, color=(0, 255, 255)):
    """
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

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image,
                    f"{cat}: {conf}",
                    (x1 - 2, y1 - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
    return image


if __name__ == "__main__":
	pass