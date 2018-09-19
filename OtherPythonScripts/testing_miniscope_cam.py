

import cv2
import time

CV_CAP_PROP_SATURATION = 12
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4

cam = cv2.VideoCapture(1)

cam.set(CV_CAP_PROP_SATURATION, 23)
#cam.set(CV_CAP_PROP_FPS, 5)



start = time.time()
frame_num = 0
while True:
    ret_val, img = cam.read()
    cv2.imshow('my webcam', img)
    frame_num += 1
    if cv2.waitKey(1) == 27:
        break  # esc to quit
dt = time.time() - start
print(frame_num/dt)
cv2.destroyAllWindows()
del cam


