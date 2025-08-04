import sklearn
import numpy as np
import os
from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
# 图像存储位置
file_root = '/root/autodl-tmp/Data/Training'
# svm训练数据
path='/root/fruit/svm_data.txt'

type_list = os.listdir(file_root)
def labels(s):
    it={b'Apple Braeburn': 0, b'Apple Crimson Snow': 1, b'Apple Golden 1': 2, b'Apple Golden 2': 3, b'Apple Golden 3': 4, b'Apple Granny Smith': 5, b'Apple Pink Lady': 6, b'Apple Red 1': 7, b'Apple Red 2': 8, b'Apple Red 3': 9, b'Apple Red Delicious': 10, b'Apple Red Yellow 1': 11, b'Apple Red Yellow 2': 12, b'Apricot': 13, b'Avocado': 14, b'Avocado ripe': 15, b'Banana': 16, b'Banana Lady Finger': 17, b'Banana Red': 18, b'Beetroot': 19, b'Blueberry': 20, b'Cactus fruit': 21, b'Cantaloupe 1': 22, b'Cantaloupe 2': 23, b'Carambula': 24, b'Cauliflower': 25, b'Cherry 1': 26, b'Cherry 2': 27, b'Cherry Rainier': 28, b'Cherry Wax Black': 29, b'Cherry Wax Red': 30, b'Cherry Wax Yellow': 31, b'Chestnut': 32, b'Clementine': 33, b'Cocos': 34, b'Corn': 35, b'Corn Husk': 36, b'Cucumber Ripe': 37, b'Cucumber Ripe 2': 38, b'Dates': 39, b'Eggplant': 40, b'Fig': 41, b'Ginger Root': 42, b'Granadilla': 43, b'Grape Blue': 44, b'Grape Pink': 45, b'Grape White': 46, b'Grape White 2': 47, b'Grape White 3': 48, b'Grape White 4': 49, b'Grapefruit Pink': 50, b'Grapefruit White': 51, b'Guava': 52, b'Hazelnut': 53, b'Huckleberry': 54, b'Kaki': 55, b'Kiwi': 56, b'Kohlrabi': 57, b'Kumquats': 58, b'Lemon': 59, b'Lemon Meyer': 60, b'Limes': 61, b'Lychee': 62, b'Mandarine': 63, b'Mango': 64, b'Mango Red': 65, b'Mangostan': 66, b'Maracuja': 67, b'Melon Piel de Sapo': 68, b'Mulberry': 69, b'Nectarine': 70, b'Nectarine Flat': 71, b'Nut Forest': 72, b'Nut Pecan': 73, b'Onion Red': 74, b'Onion Red Peeled': 75, b'Onion White': 76, b'Orange': 77, b'Papaya': 78, b'Passion Fruit': 79, b'Peach': 80, b'Peach 2': 81, b'Peach Flat': 82, b'Pear': 83, b'Pear 2': 84, b'Pear Abate': 85, b'Pear Forelle': 86, b'Pear Kaiser': 87, b'Pear Monster': 88, b'Pear Red': 89, b'Pear Stone': 90, b'Pear Williams': 91, b'Pepino': 92, b'Pepper Green': 93, b'Pepper Orange': 94, b'Pepper Red': 95, b'Pepper Yellow': 96, b'Physalis': 97, b'Physalis with Husk': 98, b'Pineapple': 99, b'Pineapple Mini': 100, b'Pitahaya Red': 101, b'Plum': 102, b'Plum 2': 103, b'Plum 3': 104, b'Pomegranate': 105, b'Pomelo Sweetie': 106, b'Potato Red': 107, b'Potato Red Washed': 108, b'Potato Sweet': 109, b'Potato White': 110, b'Quince': 111, b'Rambutan': 112, b'Raspberry': 113, b'Redcurrant': 114, b'Salak': 115, b'Strawberry': 116, b'Strawberry Wedge': 117, b'Tamarillo': 118, b'Tangelo': 119, b'Tomato 1': 120, b'Tomato 2': 121, b'Tomato 3': 122, b'Tomato 4': 123, b'Tomato Cherry Red': 124, b'Tomato Heart': 125, b'Tomato Maroon': 126, b'Tomato not Ripened': 127, b'Tomato Yellow': 128, b'Walnut': 129, b'Watermelon': 130}
    return it[s]

data=np.loadtxt(path, dtype=float, delimiter=',', converters={256:labels})
x, y = np.split(data, indices_or_sections=(256,), axis=1)  # x为数据，y为标签
train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(x, y, random_state=1,train_size=0.8,test_size=0.2)

classifier=svm.SVC(kernel='rbf')
print('载入数据')
classifier.fit(train_data,train_label.ravel()) # ravel函数在降维时默认是行序优先
print('训练完成')
s = pickle.dumps(classifier)
f = open('/root/fruit/svm.model', "wb+")
f.write(s)
f.close()
print('保存模型')
print("训练集：",classifier.score(train_data,train_label.ravel()))
print("测试集：",classifier.score(test_data,test_label.ravel()))
result = classifier.predict(test_data)
np.savetxt( "data_init.csv",test_data,fmt='%f',delimiter=',')
np.savetxt( "predict.csv",result,fmt='%f',delimiter=',')



