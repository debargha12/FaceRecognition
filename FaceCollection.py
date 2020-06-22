#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2



# In[4]:


#object of the harcascade classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[7]:


#extract features of a face
def faceExtract(img):
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converting to grayscale image
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        croppedFace=img[y:y+h,x:x+w]
    return croppedFace
    


# In[9]:


capture = cv2.VideoCapture(0)
count=0
while True:
    ret, frame = capture.read()
    if faceExtract(frame) is not None:
        count+=1
        face = cv2.resize(faceExtract(frame),(200,200))
        face= cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        filepath='img/user'+str(count)+'.jpg'
        cv2.imwrite(filepath,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper', face)
    else:
        print('Face not found')
        pass
    if cv2.waitKey(1)==13 or count ==200: #13 ascii of enter
        break
capture.release()
cv2.destroyAllWindows()
print('Samples Collected!!')
        


# In[ ]:




