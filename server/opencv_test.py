import cv2

vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
