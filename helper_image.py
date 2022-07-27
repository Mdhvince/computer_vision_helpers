import time
import cv2
import imutils
import numpy as np
import albumentations as albu
import matplotlib.pyplot as plt


def set_window_pos(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, 0, 0)


def safe_close(cap):
    cap.release()
    cv2.destroyAllWindows()


def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 640
    cap.set(4, 480)  # 480
    time.sleep(2.)
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    center = int(cols / 2)
    return cap, center, rows


def sliding_window(image, stepSize, windowSize):
    """
    :param image: RGB or BGR Image
    :param stepSize: Step size in pixel of the sliding
    :param windowSize: Window Size in pixel
    :return: yield current x1, current y1, current window (cropped image)
    """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def pad_to_reach(array, shape):
    """
    :param array: numpy array
    :param shape: tuple of int, shape to be reached by the padding
    :return:
    """
    result = np.zeros(shape, dtype=np.uint8)
    result[:array.shape[0], :array.shape[1], :] = array
    return result


def augment_images_boxes(image, bbox, category, list_augmentations, pOneOf=1, pCompose=1):
    """
    :param image: RGB or BGR Image
    :param bbox: 2D boxes list
    :param category: 1D array or list of category_id that describe the object
    :param list_augmentations: list of alumentations augmentations
    :param pOneOf: probability of selecting at least one of the list of augmentations
    :param pCompose: probability of applying the augmentation
    :return: augmented image, augmented boxes
    """
    bbox = np.round(np.array(bbox), 1)
    bbox = [b.tolist() for b in bbox]
    annotations = {'image': image, 'bboxes': bbox, 'category_id': category}
    compose = albu.Compose(
        [albu.OneOf(list_augmentations, p=pOneOf)],
        p=pCompose,
        bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_id'])
    )
    transformed = compose(**annotations)
    im = transformed['image']
    bbox = [list(i) for i in transformed['bboxes']]
    return im, bbox


def get_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    return extLeft, extRight, extTop, extBot


def show_image(image, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def get_rgb_channels(image):
    """Covert first to RGB Channel if image openned using cv2"""
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    return r, g, b


def ft_image(norm_image):
    '''This function takes in a normalized (/255.), grayscale image
       and returns a frequency spectrum transform of that image. '''
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift))
    
    return fshift, frequency_tx

def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max - min) + min
    return x

def psnr(img1, img2):
    """img1 and img2 have range [0, 255].
    to measure image similarity between an image and its reconstruction for example."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


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
