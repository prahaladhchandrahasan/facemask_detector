import numpy as np 
import cv2 
import os

CATEGORIES = ["with_mask", "without_mask"]
lookup = {"with_mask" :0, "without_mask" :1}
data = []
lables = []


for category in CATEGORIES:
	path = os.path.join("dataset", category)
	for img in os.listdir(path):
		img_path = os.path.join(path, img)
		image = cv2.imread(img_path)
		image = cv2.resize(image,(224,224))
		image = np.array(image, "float32")
		image = image/255.0
		data.append(image)
		lables.append(lookup[category])


data = np.array(data, "float32")
lables = np.array(lables,"float32")
print("[INFO] Shapes of data and lables arrays")
print(data.shape)
print(lables.shape)

np.save("data.npz",data)
np.save("lables.npy",lables)
print("[INFO] Arrays saved to disk")


