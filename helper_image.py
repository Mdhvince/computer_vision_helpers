import cv2
import imutils
import numpy as np
import albumentations as albu
import matplotlib.pyplot as plt


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
    
    return frequency_tx


if __name__ == '__main__':
    pass
