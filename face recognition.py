import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
path ='photos'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    currimg=cv2.imread(f'{path}/{cl}')
    images.append(currimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markattendance(name):
    with open('attendance.csv','r+') as f:
         mydatalist=f.readline()
         namelist=[]
         print(mydatalist)
         for line in mydatalist:
             entry=name.split(',')
             namelist.append(entry[0])
         if name  in namelist:
             now=datetime.now()
             dtstring=now.strftime('%H:%M:%S')
             f.write(f'{name},{dtstring}')
             f.write("\n")

encodeListknown = findEncodings(images)
print(len(encodeListknown))

cap = cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(img,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches= face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis= face_recognition.face_distance(encodeListknown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matchIndex in matches:
            name = classnames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)




