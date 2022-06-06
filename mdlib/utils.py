import cv2
import numpy as np


def resize2SquareKeepingAspectRation(img, size, interpolation):
    '''
    :param img: image (frame)
    :param size: resize to width: size, height: size
    :param interpolation: resize interpolation
    :return: resized frame, resize value as dict as like: x_pos, y_pos, ratio
    '''
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation), h / size
    if h > w:
        dif = h
    else:
        dif = w
    x_pos = int((dif - w) / 2.)
    y_pos = int((dif - h) / 2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation), {'x_pos': x_pos, 'y_pos': y_pos, 'ratio': dif / size}
