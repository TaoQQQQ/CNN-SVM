#D:\PycharmProjects\pythonProject\output_files\fruit-360 model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import numpy
import os
import cv2 as cv
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Lambda
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
np.set_printoptions(suppress=True)
def convert_to_hsv_and_grayscale(x):
    hsv = tf.image.rgb_to_hsv(x)#调节图像的灰度值
    gray = tf.image.rgb_to_grayscale(x) #将rgb格式的图像转换成grayscale的灰度图像
    rez = tf.concat([hsv, gray], axis=-1)#连接第一个维度
    return rez

model=load_model('D:\\PycharmProjects\\pythonProject\\output_files\\fruit-360 model\\model.h5')
fcl2_layer_model = Model(inputs=model.input,outputs=model.get_layer('fcl2').output)
f=open('D:\PycharmProjects\pythonProject\svm_data.txt','a')
file_root = 'D:\\deep learing\\Fruit-Images-Dataset-master\\Fruit-Images-Dataset-master\\Test'  # 当前文件夹下的所有图片
type_list = os.listdir(file_root)
for type_name in type_list:
    type_path= file_root+'\\'+type_name
    img_list = os.listdir(type_path)
    for img_name in img_list:
      img_path = type_path +'\\'+img_name
      img = cv.imread(img_path)
      img = img.astype(dtype='float32')
      img = tf.expand_dims(img,0)
      output = fcl2_layer_model.predict(img)
      output = tf.squeeze(output,[0])
      output =numpy.array(output)
      tem_str = ','.join(str(i) for i in output)
      tem_str = tem_str + ',' + type_name + '\n'
      f.write(tem_str)
print('完成')
f.close()
