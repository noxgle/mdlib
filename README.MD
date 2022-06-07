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
if MD.check() is True:
    print('Move detect')
```

## Move detection
### Init function class

        :param resize_to_size: resize image (frame) before motion detect for lower CPU usage, default: 100px
        :param resize_interpolation: resize interpolation, default: cv2.INTER_NEAREST
        :param gaussian_blur_ksize: ksize for cv2.GaussianBlur, default: (5, 5)
        :param gaussian_blur_sigmax: sigmaX for cv2.GaussianBlur, default: 255
        :param dilate_kernel: kernel for cv2.dilate, default: (5, 5)
        :param threshold_tresh: thresh for cv2.threshold, default: 20
        :param threshold_maxval: maxval for cv2.threshold, default 255
        :param contour_size_detection: the minimum size of the object that must be detected, default: 50
        :param keep_ascpect_ratio: downsize image with keep aspect ratio, default: True

