import cv2
import numpy as np 
from keras.preprocessing.image import img_to_array
import os
from numpy import loadtxt
from keras.models import load_model

treshold = 0.7
model=load_model('model.h5')
print(model.summary())
font = cv2.FONT_HERSHEY_SIMPLEX 
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	img = cv2.resize(frame,(224,224))
	img = img_to_array(img)/255.0
	img = img.reshape((1,224,224,3))
	Y = model.predict(img)
	print(Y)
	if Y>treshold:
		pred = 'No Mask'
		cv2.putText(frame, pred, (150,400), font , 2, (0, 0, 255),3, cv2.LINE_AA)
	else:
		pred = 'Mask'
		cv2.putText(frame, pred, (150,400), font , 2, (0, 255, 0),3, cv2.LINE_AA)

	cv2.imshow('frame', frame)


	if cv2.waitKey(25) & 0xFF == ord(' '):
		break

cap.release()
cv2.destroyAllWindows()