import cv2
import time
import numpy as np
from mdlib.utils import resize2SquareKeepingAspectRation


class MoveDetection:
    '''
    MoveDetect is used to detect motion on resized image (frame).
    '''

    def __init__(self, resize_to_size=100, resize_interpolation=cv2.INTER_NEAREST, gaussian_blur_ksize=(5, 5),
                 gaussian_blur_sigmax=0, dilate_kernel=(5, 5), threshold_tresh=20, threshold_maxval=255,
                 contour_size_detection=50, keep_ascpect_ratio=True):
        '''
        Init class.

        :param resize_to_size: resize image (frame) before motion detect for lower CPU usage, default: 100px
        :param resize_interpolation: resize interpolation, default: cv2.INTER_NEAREST
        :param gaussian_blur_ksize: ksize for cv2.GaussianBlur, default: (5, 5)
        :param gaussian_blur_sigmax: sigmaX for cv2.GaussianBlur, default: 255
        :param dilate_kernel: kernel for cv2.dilate, default: (5, 5)
        :param threshold_tresh: thresh for cv2.threshold, default: 20
        :param threshold_maxval: maxval for cv2.threshold, default 255
        :param contour_size_detection: the minimum size of the object that must be detected, default: 50
        :param keep_ascpect_ratio: downsize image with keep aspect ratio, default: True
        '''

        self.resize_to_size = resize_to_size
        self.resize_interpolation = resize_interpolation
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.gaussian_blur_sigmax = gaussian_blur_sigmax
        self.dilate_kernel = dilate_kernel
        self.threshold_tresh = threshold_tresh
        self.threshold_maxval = threshold_maxval
        self.contour_size_detection = contour_size_detection
        self.keep_ascpect_ratio = keep_ascpect_ratio

        self.previous_frame = None
        self.frame = None
        self.resize_value = None
        self.contours = None
        self.last_detect_time = None

    def check(self, new_frame):
        '''
        Check if motion has been detected.

        :param new_frame: image (frame)
        :return: True if detected object or False if not
        '''
        self.contours = []
        if self.keep_ascpect_ratio is True:
            self.frame, self.resize_value = resize2SquareKeepingAspectRation(new_frame, self.resize_to_size,
                                                                             self.resize_interpolation)
        else:
            self.frame = cv2.resize(new_frame, (self.resize_to_size, self.resize_to_size), self.resize_interpolation)

        img_brg = np.array(self.frame)
        img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

        prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=self.gaussian_blur_ksize,
                                          sigmaX=self.gaussian_blur_sigmax)

        if self.previous_frame is None:
            self.previous_frame = prepared_frame
            return False

        diff_frame = cv2.absdiff(src1=self.previous_frame, src2=prepared_frame)
        self.previous_frame = prepared_frame

        kernel = np.ones(self.dilate_kernel)
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        thresh_frame = cv2.threshold(src=diff_frame, thresh=self.threshold_tresh, maxval=self.threshold_maxval,
                                     type=cv2.THRESH_BINARY)[1]

        self.contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        for contour in self.contours:
            if cv2.contourArea(contour) < self.contour_size_detection:
                continue
            self.last_detect_time = time.time()
            return True
        return False

    def get_original_contour(self, frame):
        '''
        Get the coordinates of the detected motion for the original image.

        :param frame: image (frame)
        :return: contours for detected object
        '''
        original_contours = []
        if self.keep_ascpect_ratio is True:
            ratio = self.resize_value['ratio']
            x_pos = self.resize_value['x_pos']
            y_pos = self.resize_value['y_pos']
            for c in self.contours:
                (x, y, w, h) = cv2.boundingRect(c)
                x = int(x * ratio - x_pos)
                y = int(y * ratio - y_pos)
                w = int(w * ratio)
                h = int(h * ratio)
                original_contours.append([x, y, w, h])
            return original_contours
        else:
            ratio_x = frame.shape[1] / self.resize_to_size
            ratio_y = frame.shape[0] / self.resize_to_size
            for c in self.contours:
                (x, y, w, h) = cv2.boundingRect(c)
                x = int(x * ratio_x)
                y = int(y * ratio_y)
                w = int(w * ratio_x)
                h = int(h * ratio_y)
                original_contours.append([x, y, w, h])
            return original_contours.append([x, y, w, h])

    def get_resized_contour(self):
        '''
        Get the coordinates of the detected motion for the resized image.

        :return: contours for detected object
        '''
        resized_contours = []
        for c in self.contours:
            (x, y, w, h) = cv2.boundingRect(c)
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            resized_contours.append([x, y, w, h])
        return resized_contours

    def draw_contour_on_original_frame(self, frame, color=(0, 255, 0), thickness=1):
        '''
        Draw contours on original frame.

        :param frame: image (frame)
        :param color: rectangle color, default: (0, 255, 0)
        :param thickness: rectangle thickness, default 1
        :return: image (frame)
        '''
        if self.keep_ascpect_ratio is True:
            ratio = self.resize_value['ratio']
            x_pos = self.resize_value['x_pos']
            y_pos = self.resize_value['y_pos']
            for c in self.contours:
                (x, y, w, h) = cv2.boundingRect(c)
                x = int(x * ratio - x_pos)
                y = int(y * ratio - y_pos)
                w = int(w * ratio)
                h = int(h * ratio)
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=thickness)
            return frame
        else:
            ratio_x = frame.shape[1] / self.resize_to_size
            ratio_y = frame.shape[0] / self.resize_to_size
            for c in self.contours:
                (x, y, w, h) = cv2.boundingRect(c)
                x = int(x * ratio_x)
                y = int(y * ratio_y)
                w = int(w * ratio_x)
                h = int(h * ratio_y)
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=thickness)
            return frame

    def draw_contour_on_resized_frame(self, color=(0, 255, 0), thickness=1):
        '''
        Draw contours on resized frame.

        :param frame: image (frame)
        :param color: rectangle color, default: (0, 255, 0)
        :param thickness: rectangle thickness, default: 1
        :return: image (frame)
        '''
        for c in self.contours:
            (x, y, w, h) = cv2.boundingRect(c)
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            cv2.rectangle(self.frame, pt1=(x, y), pt2=(x + w, y + h), color=color, thickness=thickness)
        return self.frame
