import os
from cProfile import label
from statistics import mode

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

face_data = []

dic={
    0: 'adi',
    1: 'amitabh',
    2: 'saniya'
}

labels = []
inx = 0

for file in os.listdir("project/data"):
    data = np.load(f"./project/data/{file}")
    face_data.append(data)
    l = [inx for i in range(data.shape[0])]
    labels.extend(l)
    inx+=1

X = np.concatenate(face_data,axis=0)
Y = np.array(labels).reshape(-1,1)

print(X.shape,Y.shape)

knn = KNeighborsClassifier()
knn.fit(X,Y)

cam= cv2.VideoCapture(0)
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read() #takes x,y,h,w of the face

    if ret == False:
        continue
    
    gray_face = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #coverting color frames to grey frames
    faces = model.detectMultiScale(gray_face,1.1,5)

    if len(faces) == 0:
        continue

    for face in faces:
        #extract the face from the frame
        x,y,w,h = face 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        offset = 5
        face_selection = gray_face[y-offset:y+h+offset,x-offset:x+w+offset]
        face_selection = cv2.resize(face_selection,(100,100))

        query = face_selection.reshape(-1,10000)

        #predicition from the model
        pred = knn.predict(query)[0]

        name = dic[int(pred)]

        cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,255)
        cv2.imshow("My video", frame)
    
    key_press = cv2.waitKey(1) & 0xFF
    
    if key_press == 113:
        break

cam.release()
cv2.destroyAllWindows()


