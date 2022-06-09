import cv2
import time
import numpy as np
from mdlib.utils import resize2SquareKeepingAspectRation


class MoveDetection:
    """
    MoveDetect is used to detect motion on resized image (frame).
    """

    def __init__(self, resize_to_size=100, resize_interpolation=cv2.INTER_NEAREST, gaussian_blur_ksize=(5, 5),
                 gaussian_blur_sigmax=0, dilate_kernel=(5, 5), threshold_tresh=20, threshold_maxval=255,
                 contour_size_detection=50, keep_ascpect_ratio=True):
        """
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
        """

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
        """
        Check if motion has been detected.

        :param new_frame: image (frame)
        :return: True if detected object or False if not
        """
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

    def get_original(self, frame):
        """
        Get the rectangles data of the detected motion for the original image.

        :param frame: image (frame)
        :return: [[x,y,w,h]] for detected object
        """
        rect = []
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
                rect.append([x, y, w, h])
            return rect
        else:
            ratio_x = frame.shape[1] / self.resize_to_size
            ratio_y = frame.shape[0] / self.resize_to_size
            for c in self.contours:
                (x, y, w, h) = cv2.boundingRect(c)
                x = int(x * ratio_x)
                y = int(y * ratio_y)
                w = int(w * ratio_x)
                h = int(h * ratio_y)
                rect.append([x, y, w, h])
            return rect

    def get_original_combined(self, frame):
        """
        Get the one combined rectangle data of the detected motion for the original image.

        :param frame: image (frame)
        :return: [[x,y,w,h]] for detected object
        """
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []

        for c in self.get_original(frame):
            x1_list.append(c[0])
            y1_list.append(c[1])
            x2_list.append(c[2] + c[0])
            y2_list.append(c[3] + c[1])

        if len(x1_list) > 0 and len(y1_list) > 0 and len(x2_list) > 0 and len(y2_list) > 0:
            x = min(x1_list)
            y = min(y1_list)
            w = max(x2_list) - x
            h = max(y2_list) - y
            return [[x, y, w, h]]
        else:
            return []

    def get_resize(self):
        """
        Get the rectangles data of the detected motion for the resized image.

        :return: [[x,y,w,h]] for detected object
        """
        rect_coordinates = []
        for c in self.contours:
            (x, y, w, h) = cv2.boundingRect(c)
            x1 = int(x)
            y1 = int(y)
            w = int(w)
            h = int(h)
            rect_coordinates.append([x1, y1, w, h])
        return rect_coordinates

    def get_resize_combined(self, frame):
        """
        Get the one combined rectangle data of the detected motion for the resized image.

        :param frame: image (frame)
        :return: [[x,y,w,h]] for detected object
        """
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []

        for c in self.get_resize():
            x1_list.append(c[0])
            y1_list.append(c[1])
            x2_list.append(c[2] + c[0])
            y2_list.append(c[3] + c[1])

        if len(x1_list) > 0 and len(y1_list) > 0 and len(x2_list) > 0 and len(y2_list) > 0:
            x = min(x1_list)
            y = min(y1_list)
            w = max(x2_list) - x
            h = max(y2_list) - y
            return [[x, y, w, h]]
        else:
            return []

    def draw_original(self, frame, color=(0, 255, 0), thickness=1, combined=False):
        """
        Draw a rectangle in the area where motion has been detected on original frame.

        :param frame: original image (frame)
        :param color: rectangle color, default: (0, 255, 0)
        :param thickness: rectangle thickness, default: 1
        :param combined: if True return one combined rectangle data, default: False
        :return: image (frame)
        """

        if combined is True:
            for r in self.get_original_combined(frame):
                cv2.rectangle(frame, pt1=(r[0], r[1]), pt2=(r[0] + r[2], r[1] + r[3]), color=color, thickness=thickness)
        else:
            for r in self.get_original(frame):
                cv2.rectangle(frame, pt1=(r[0], r[1]), pt2=(r[0] + r[2], r[1] + r[3]), color=color, thickness=thickness)

        return frame

    def draw_resized(self, color=(0, 255, 0), thickness=1, combined=False):
        """
        Draw a rectangle in the area where motion has been detected on resized frame.

        :param color: rectangle color, default: (0, 255, 0)
        :param thickness: rectangle thickness, default: 1
        :param combined: if True return one combined rectangle data, default: False
        :return: image (frame)
        """
        if combined is True:
            for r in self.get_resize_combined():
                cv2.rectangle(self.frame, pt1=(r[0], r[1]), pt2=(r[0] + r[2], r[1] + r[3]), color=color,
                              thickness=thickness)
        else:
            for r in self.get_resize():
                cv2.rectangle(self.frame, pt1=(r[0], r[1]), pt2=(r[0] + r[2], r[1] + r[3]), color=color,
                              thickness=thickness)

        return self.frame
