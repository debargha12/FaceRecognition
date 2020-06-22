import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


datalocation= 'img/'
images =[file for file in listdir(datalocation)if isfile(join(datalocation,file))]
TrainData, labels =[], []
for i,files in enumerate(images):
    imagePath = datalocation +images[i]
    imageRead= cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    TrainData.append(np.asarray(imageRead, dtype= np.uint8))
    labels.append(i)

labels = np.asarray(labels,dtype=np.int32)
model= cv2.face.LBPHFaceRecognizer_create()  #linear binary face histogram
model.train(np.asarray(TrainData),np.asarray(labels))
print('Model training done')

model.save('model.xml')

