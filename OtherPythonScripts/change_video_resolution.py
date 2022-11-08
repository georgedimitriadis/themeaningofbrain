
import cv2
import numpy as np

cap = cv2.VideoCapture(r'D:\\AK_33.1\2018_04_30-11_38\Analysis\NNs\BonsaiCropping\Full_video.avi')

fourcc = cv2.VideoWriter_fourcc(*'FMP4')
resolution = (150, 112)
fps = 120
out = cv2.VideoWriter(r'D:\\AK_33.1\2018_04_30-11_38\Analysis\NNs\SubsampledVideo\Video_undistrorted_150x112_120fps.mp4',
                      fourcc, fps, resolution)

f = 0
while True:
    ret, frame = cap.read()
    if ret==True:
        frame = frame[64:576, :, :]
        b = cv2.resize(frame, resolution, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        out.write(b)
        f += 1
    else:
        break

    if f % 1000 == 0:
        print(f)

cap.release()
out.release()
cv2.destroyAllWindows()
