import tensorflow as tf
import cv2
import numpy as np
import os
from PIL.Image import fromarray

class NormalizeTo01(object):
    def __call__(self, tensor):
        return (tensor + 1) / 2
    
PATH = "C:\\Users\\User\\Desktop\\estr3114\\project\\source\\number\\archive\\trainingSet\\trainingSet"
def data_processing():
    for _,__,files in os.walk(PATH):
        return files

#image resize from 28*28 to 64*64
def image_resize(img):
    img = cv2.resize(img, (64,64))
    return img

FILES = data_processing()
def image_data(start,end):
    global Images
    Images = []
    for id, file in enumerate(FILES[start:end]):
        img = cv2.imread(os.path.join(PATH,file))
        img = image_resize(img)
        img = tf.keras.utils.img_to_array(fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
        img = (img-127.5)/127.5
        Images.append(img)
    Images = np.array(Images)
    return Images

for id in range(0, len(FILES), 1600):
    train_doc = image_data(id,id+1600)
    np.save(f"C:\\Users\\User\\Desktop\\estr3114\\project\\source\\image_\\train\\number\\number_{id+1600}.npy",train_doc)
