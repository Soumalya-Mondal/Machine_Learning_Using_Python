import cv2
import tensorflow as tf
CATAGORIES=['Human','Horse']

def prepare(filepath):
	img_size=100
	img_arr=cv2.imread(filepath)
	img_arr=cv2.resize(img_arr,(img_size,img_size))
	return img_arr.reshape(-1,img_size,img_size,3)

model=tf.keras.models.load_model('my_model.h5')
prediction=model.predict([prepare('soumalya.png')])
print(CATAGORIES[int(prediction[0][0])])