# mdlib

Mdlib is used to detect motion on resized image (frame).

## Inataltion

At the moment, the installation looks like this :

* pip install git+https://github.com/noxgle/mdlib.git

### Installation dependencies

* OpenCV

### Test

Checkout test_move_detection.py in test folder.

### Example usage

```python
import cv2
from mdlib import MoveDetection

cap = cv2.VideoCapture(0)
MD = MoveDetection()
while True:
    ret, frame = cap.read()
    if MD.check(frame) is True:
        print('Move detect')
```

## Class and methods description

### MoveDetection()

#### init

        All parm are predefined.    

        :param resize_to_size: resize image (frame) before motion detect for lower CPU usage, default: 100px
        :param resize_interpolation: resize interpolation, default: cv2.INTER_NEAREST
        :param gaussian_blur_ksize: ksize for cv2.GaussianBlur, default: (5, 5)
        :param gaussian_blur_sigmax: sigmaX for cv2.GaussianBlur, default: 255
        :param dilate_kernel: kernel for cv2.dilate, default: (5, 5)
        :param threshold_tresh: thresh for cv2.threshold, default: 20
        :param threshold_maxval: maxval for cv2.threshold, default 255
        :param contour_size_detection: the minimum size of the object that must be detected, default: 50
        :param keep_ascpect_ratio: downsize image with keep aspect ratio, default: True

#### check

        Check if motion has been detected.

        :param new_frame: image (frame)
        :return: True if detected object or False if not

#### get_original

        Get the rectangles data of the detected motion for the original image.

        :param frame: image (frame)
        :return: [[x,y,w,h]] for detected object

#### get_original_combined

        Get the one combined rectangle data of the detected motion for the original image.

        :param frame: image (frame)
        :return: [[x,y,w,h]] for detected object

#### get_resize

        Get the rectangles data of the detected motion for the resized image.

        :return: [[x,y,w,h]] for detected object

#### get_resize_combined

        Get the one combined rectangle data of the detected motion for the resized image.

        :return: [[x,y,w,h]] for detected object

#### draw_original

        Draw a rectangle in the area where motion has been detected on original frame.

        :param frame: original image (frame)
        :param color: rectangle color, default: (0, 255, 0)
        :param thickness: rectangle thickness, default: 1
        :param combined: if True return one combined rectangle data, default: False
        :return: image (frame)

#### draw_resized

        Draw a rectangle in the area where motion has been detected on resized frame.
        
        :param frame: resized image (frame)
        :param color: rectangle color, default: (0, 255, 0)
        :param thickness: rectangle thickness, default: 1
        :param combined: if True return one combined rectangle data, default: False
        :return: image (frame)

Shorten link with: https://lvk.pl
