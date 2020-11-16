import numpy as np
import cv2
import os
import random
import pickle
dir_dataset=r'C:\Users\Soumalya\Desktop\ML_H_v_H\dataset'
CATEGORIES=['horse','human']


img_size=100

data=[]


for catagory in CATEGORIES:
	folder=os.path.join(dir_dataset,catagory)
	label=CATEGORIES.index(catagory)
	for img in os.listdir(folder):
		img_path=os.path.join(folder,img)
		img_arr=cv2.imread(img_path)
		img_arr=cv2.resize(img_arr,(img_size,img_size))
		data.append([img_arr,label])
random.shuffle(data)
X=[]
y=[]
for features,label in data:
	X.append(features)
	y.append(label)
X=np.array(X)
y=np.array(y)
pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))