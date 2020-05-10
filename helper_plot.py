import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def basis_plotting_style(title, xlabel, ylabel, rotation_xlabel, rotation_ylabel):
    plt.style.use('seaborn-paper')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation_xlabel)
    plt.yticks(rotation=rotation_ylabel)


def draw_arrow_on_plot(text, x_pos_arrow, y_pos_arrow):
    config_arrow = dict(headwidth=7, facecolor='black', shrink=0.05, width=2)
    plt.annotate(text, xy=(x_pos_arrow, y_pos_arrow), xycoords='data',
                 xytext=(0.75, 0.95), textcoords='axes fraction',
                 arrowprops=config_arrow,
                 horizontalalignment='right', verticalalignment='top',
                 )



if __name__ == '__main__':
    pass