import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Lambda
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

#tf.test.is_gpu_available()
##############################################
learning_rate = 0.1  # initial learning rate
min_learning_rate = 0.00001  # once the learning rate reaches this value, do not decrease it further
learning_rate_reduction_factor = 0.5  # the factor used when reducing the learning rate -> learning_rate *= learning_rate_reduction_factor
patience = 3  # how many epochs to wait before reducing the learning rate when the loss plateaus
verbose = 1  # controls the amount of logging done during training and testing: 0 - none, 1 - reports metrics after each batch, 2 - reports metrics after each epoch
image_size = (100, 100)  # width and height of the used images
input_shape = (100, 100, 3)  # the expected input shape for the trained models; since the images in the Fruit-360 are 100 x 100 RGB images, this is the required input shape

use_label_file = False  # set this to true if you want load the label names from a file; uses the label_file defined below; the file should contain the names of the used labels, each label on a separate line
label_file = 'labels.txt'
base_dir = '/root/autodl-tmp/Data'  # relative path to the Fruit-Images-Dataset folder
test_dir = os.path.join(base_dir, 'Test')
train_dir = os.path.join(base_dir, 'Training')
output_dir = '/root/autodl-tmp/output_files'  # root folder in which to save the the output files; the files will be under output_files/model_name
##############################################
#默认在当前目录下寻找，没有就创造一个
if not os.path.exists(output_dir):
    os.makedirs(output_dir)#创造目录名

# if we want to train the network for a subset of the fruit classes instead of all, we can set the use_label_file to true and place in the label_file the classes we want to train for, one per line
if use_label_file:
    with open(label_file, "r") as f:
        labels = [x.strip() for x in f.readlines()]#就是自己设置类别
else:
    labels = os.listdir(train_dir)#返回一个由文件名和目录名组成的列表
num_classes = len(labels)


# create 2 charts, one for accuracy, one for loss, to show the evolution of these two metrics during the training process
def plot_model_history(model_history, out_path=""):#就画图的作用
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    #设置刻度
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']))
    #名字
    axs[0].legend(['train'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']))
    axs[1].legend(['train'], loc='best')
    # save the graph in a file called "acc.png" to be available for later; the model_name is provided when creating and training a model
    if out_path:
        plt.savefig(out_path + "/acc.png")
    plt.show()


# create a confusion matrix to visually represent incorrectly classified images
#创建混淆矩阵，直观地表示分类错误的图像
def plot_confusion_matrix(y_true, y_pred, classes, out_path=""):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(40, 40))
    ax = sn.heatmap(df_cm, annot=True, square=True, fmt="d", linewidths=.2, cbar_kws={"shrink": 0.8})
    if out_path:
        plt.savefig(out_path + "/confusion_matrix.png")  # as in the plot_model_history, the matrix is saved in a file called "model_name_confusion_matrix.png"
    return ax


# Randomly changes hue and saturation of the image to simulate variable lighting conditions
#随机改变图像的色调和饱和度以模拟可变的光照条件
def augment_image(x):
    x = tf.image.random_saturation(x, 0.9, 1.2)#随机调整饱和度
    x = tf.image.random_hue(x, 0.02)
    return x


# given the train and test folder paths and a validation to test ratio, this method creates three generators
#  - the training generator uses (100 - validation_percent) of images from the train set
#    it applies random horizontal and vertical flips for data augmentation and generates batches randomly
#  - the validation generator uses the remaining validation_percent of images from the train set
#    does not generate random batches, as the model is not trained on this data
#    the accuracy and loss are monitored using the validation data so that the learning rate can be updated if the model hits a local optimum
#  - the test generator uses the test set without any form of augmentation
#    once the training process is done, the final values of accuracy and loss are calculated on this set

#给定train和test文件夹路径以及验证与测试比率，该方法将创建三个生成器

#-训练生成器使用来自训练集的（100-100%）图像

#它应用随机水平和垂直翻转来扩充数据，并随机生成批

#-验证生成器使用训练组图像的剩余验证百分比

#不会生成随机批次，因为模型未对此数据进行训练

#使用验证数据监控精度和损失，以便在模型达到局部最优时更新学习率

#-测试生成器使用测试集而不进行任何形式的扩充

#一旦训练过程完成，在这个集合上计算准确度和损失的最终值

#就是把图像从文件拿出来处理一下，把训练数据挪一挪啥的

#keras数据扩充作用来增加训练数据

def build_data_generators(train_folder, test_folder, labels=None, image_size=(100, 100), batch_size=50):
    # 指定参数
    # rotation_range 旋转
    # width_shift_range 左右平移
    # height_shift_range 上下平移
    # zoom_range 随机放大或缩小
    #vertical_flip：布尔值，进行随机竖直翻转
    train_datagen = ImageDataGenerator(
        width_shift_range=0.0,
        height_shift_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,  # randomly flip images
        preprocessing_function=augment_image)
# augmentation is done only on the train set (and optionally validation)

#preprocessing_function: 将被应用于每个输入的函数。该函数将在图片缩放和数据提升之后运行。

# 该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array

#preprocessing_function: 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。
    test_datagen = ImageDataGenerator()
    #地址 size 返回标签数组类型（1D整数标签  一批数据的大小  是否混洗数据    subset: 数据子集 ("training" 或 "validation")      classes: 可选的类的子目录列表（例如 ['dogs', 'cats']） target_size:是否打乱数据
    train_gen = train_datagen.flow_from_directory(train_folder, target_size=image_size, class_mode='sparse',
                                                  batch_size=batch_size, shuffle=True, subset='training', classes=labels)
    test_gen = test_datagen.flow_from_directory(test_folder, target_size=image_size, class_mode='sparse',
                                                batch_size=batch_size, shuffle=False, subset=None, classes=labels)
    return train_gen, test_gen
#返回值：

#一个生成(x, y)元组的 DirectoryIterator，其中 x 是一个包含一批尺寸为 (batch_size, *target_size, channels)的图像的 Numpy 数组，

# y 是对应标签的 Numpy 数组。

# Create a custom layer that converts the original image from
# RGB to HSV and grayscale and concatenates the results
# forming in input of size 100 x 100 x 4

#创建一个自定义层，将原始图像从

#RGB到HSV和灰度并连接结果

#尺寸为100 x 100 x 4的输入成型

def convert_to_hsv_and_grayscale(x):
    hsv = tf.image.rgb_to_hsv(x)#调节图像的灰度值
    gray = tf.image.rgb_to_grayscale(x) #将rgb格式的图像转换成grayscale的灰度图像
    rez = tf.concat([hsv, gray], axis=-1)#连接第一个维度
    return rez

#对于axis等于负数的情况

#负数在数组索引里面表示倒数(countdown)。比如，

#对于列表ls = [1,2,3]而言，ls[-1] = 3，表示读取倒数第一个索引对应值。

#padding 填充方式，valid舍弃，same补0

def network(input_shape, num_classes):
    img_input = tf.keras.Input(shape=input_shape, name='data')
    #lambda是python的一个函数，就经过这个函数处理
    x = Lambda(convert_to_hsv_and_grayscale)(img_input)
    x = Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool1')(x)
    x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv2')(x)
    x = Activation('relu', name='conv2_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool2')(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same', name='conv3')(x)
    x = Activation('relu', name='conv3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool3')(x)
    x = Conv2D(128, (5, 5), strides=(1, 1), padding='same', name='conv4')(x)
    x = Activation('relu', name='conv4_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool4')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='fcl1')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', name='fcl2')(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation='softmax', name='predictions')(x)
    rez = Model(inputs=img_input, outputs=out)
    return rez


# this method performs all the steps from data setup, training and testing the model and plotting the results
# the model is any trainable model; the input shape and output number of classes is dependant on the dataset used, in this case the input is 100x100 RGB images and the output is a softmax layer with 118 probabilities
# the name is used to save the classification report containing the f1 score of the model, the plots showing the loss and accuracy and the confusion matrix
# the batch size is used to determine the number of images passed through the network at once, the number of steps per epochs is derived from this as (total number of images in set // batch size) + 1

#该方法完成了从数据建立、模型训练和测试到结果绘制的所有步骤

#模型是任何可训练模型；类的输入形状和输出数量取决于所使用的数据集，在这种情况下，输入是100x100 RGB图像，输出是具有118个概率的softmax层

#该名称用于保存分类报告，其中包含模型的f1分数、显示损失和精度的曲线图以及混淆矩阵

#批大小用于确定一次通过网络的图像数，每个历元的步数由此导出为（set//batch size中的图像总数）+1

def train_and_evaluate_model(model, name="", epochs=25, batch_size=50, verbose=verbose, useCkpt=False):
    print(model.summary())
    model_out_dir = os.path.join(output_dir, name)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    if useCkpt:
        model.load_weights(model_out_dir + "/model.h5")

    trainGen, testGen = build_data_generators(train_dir, test_dir, labels=labels, image_size=image_size, batch_size=batch_size)
    optimizer = Adadelta(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=patience, verbose=verbose,
                                                factor=learning_rate_reduction_factor, min_lr=min_learning_rate)
    save_model = ModelCheckpoint(filepath=model_out_dir + "/model.h5", monitor='loss', verbose=verbose,
                                 save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
    #保存最好的模型
    #steps_per_rpoch的作用就是，如果最后剩下一点点图，不够一个新的batch_size，就用这剩下的再来一次训练
    history = model.fit(trainGen,
                        epochs=epochs,
                        steps_per_epoch=(trainGen.n // batch_size) + 1,
                        verbose=verbose,
                        callbacks=[learning_rate_reduction, save_model])

    model.load_weights(model_out_dir + "/model.h5")

    trainGen.reset()#销毁计算图，提高训练速度
    #评估正确率
    #model.evaluate 用于评估您训练的模型。它的输出是准确度或损失，而不是对输入数据的预测。
    #model.predict 实际预测，其输出是目标值，根据输入数据预测。
    loss_t, accuracy_t = model.evaluate(trainGen, steps=(trainGen.n // batch_size) + 1, verbose=verbose)
    loss, accuracy = model.evaluate(testGen, steps=(testGen.n // batch_size) + 1, verbose=verbose)
    print("Train: accuracy = %f  ;  loss_v = %f" % (accuracy_t, loss_t))
    print("Test: accuracy = %f  ;  loss_v = %f" % (accuracy, loss))
    plot_model_history(history, out_path=model_out_dir)
    testGen.reset()
    #具体数据
    #//向下取整
    y_pred = model.predict(testGen, steps=(testGen.n // batch_size) + 1, verbose=verbose)
    y_true = testGen.classes[testGen.index_array]#标签
    plot_confusion_matrix(y_true, y_pred.argmax(axis=-1), labels, out_path=model_out_dir)
    class_report = classification_report(y_true, y_pred.argmax(axis=-1), target_names=labels)

    with open(model_out_dir + "/classification_report.txt", "w") as text_file:
        text_file.write("%s" % class_report)


print(labels)
print(num_classes)
model = network(input_shape=input_shape, num_classes=num_classes)
train_and_evaluate_model(model, name="fruit-360 model")
