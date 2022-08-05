from statistics import mode
import cv2


cam= cv2.VideoCapture(0)
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()

    if ret == False:
        continue

    faces = model.detectMultiScale(frame)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    cv2.imshow("My video", frame)
    key_press = cv2.waitKey(1) & 0xFF
    
    if key_press == 113:
        break

cam.release()
cv2.destroyAllWindows()