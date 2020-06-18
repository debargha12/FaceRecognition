import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle

model = cv2.face.LBPHFaceRecognizer_create()
model.read('model.xml')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def faceDetector(img, size=0.5):
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grayScale, 1.3,5)#scaling factor and neighbor
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        region = img[y:y+h, x:x+w]
        region = cv2.resize(region,(200,200))
    return img, region
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    image,face= faceDetector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result =model.predict(face)
        if result[1]< 500:
            confidence = int((1-((result[1])/300))*100)
            displayString = str(confidence)+ '%'
        cv2.putText(image, displayString,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        if confidence >75:
            cv2.putText(image, "Match ", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
        else:
            cv2.putText(image, "No Match ", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except:#no face in screen
        cv2.putText(image, "Face Not Found ", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        #cv2.putText(image, "No Match ", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Cropper', image)
        pass
    if cv2.waitKey(1)==13:
        break
capture.release()
cv2.destroyAllWindows()
