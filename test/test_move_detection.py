import cv2
from mdlib import MoveDetection

cap = cv2.VideoCapture(0)
MD = MoveDetection()
while True:
    ret, frame = cap.read()
    move_status = MD.check(frame)
    #print(frame.shape)
    if move_status is True:
        frame_o = MD.draw_contour_on_original_frame(frame)
        frame_r = MD.draw_contour_on_resized_frame()
    else:
        frame_o = frame
        frame_r = MD.frame
    cv2.imshow('orginal', frame_o)
    cv2.waitKey(1)
    cv2.imshow('resized', frame_r)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()