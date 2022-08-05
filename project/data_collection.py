from statistics import mode
import cv2


cam= cv2.VideoCapture(0)
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_data = []

skip = 0
while True:
    ret, frame = cam.read() #takes x,y,h,w of the face

    if ret == False:
        continue
    
    gray_face = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #coverting color frames to grey frames
    faces = model.detectMultiScale(gray_face,1.1,5)

    if len(faces) == 0:
        continue

    faces = sorted(faces, key= lambda f:f[2]*f[3]) #multiplying height(f[2])*widht(f[3])

    x,y,w,h = faces[-1] #rtaking only the last face with highest area into consideration 
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    offset = 5
    face_selection = gray_face[y-offset:y+h+offset,x-offset:x+w+offset]
    face_selection = cv2.resize(face_selection,(100,100))

    skip+=1 #to take pause btween capturing frames to reduce ambiguity

    if skip%10 == 0:
        face_data.append(face_selection)
        print(len(face_data))
        
    cv2.imshow("Face selection",face_selection)
    cv2.imshow("My video", frame)
    key_press = cv2.waitKey(1) & 0xFF
    
    if key_press == 113:
        break

cam.release()
cv2.destroyAllWindows()