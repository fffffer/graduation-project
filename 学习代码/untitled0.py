# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:51:26 2020

@author: djasl
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

## 获取数据集 fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## 每个图像都映射到一个标签。由于类名不包含在数据集中，因此将他们存储在此处供以后在绘制
## 图像时使用

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

# 神经网络的基本组成部分是层，图层从输入到其中的输入提取表示，深度学习的大部分内容是将简单的
# 曾链接到一起。大多层（如tf.keras.layers.Dense）具有在训练中学习的参数
# 该网络的第一层tf.keras.layers.Flatten将图像格式从二维数组（28 x 28像素）转换为一维数组（28 * 28 = #784像素）。可以将这一层看作是堆叠图像中的像素行并对齐它们。该层没有学习参数。它只会重新格式化数据。 
#像素变平后，网络由tf.keras.layers.Dense两层序列组成。这些是紧密连接或完全连接的神经层。第一Dense层有128个节点（或神经元）。第二层（也是最后一层）是一个10节点的softmax层，该层返回10个总和为1的概率分数的数组。每个节点都包含一个分数，该分数指示当前图像属于10类之一的概率。  
#  
#  


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 编译模型，在准备模型之前，需要进行一些其他设置：
#   损失函数：衡量训练过程中模型的准确性
#   优化器：基于模型看到的数据一起损失函数来更新模型的方式
#   指标：用于监视培训和测试步骤

# 训练神经网络模型需要执行以下步骤：
#   1.将训练数据输入模型
#   2.该模型学习关联图像和标签
#   3.要求模型对测试集进行预测
#   4.验证预测是否与test_labels 列阵中的标签匹配

model.fit(train_images, train_labels, epochs=10)
# 开始训练时，调用该方法

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy', test_acc)
# 比较模型在测试数据集上的表现

#--------------
# 做出预测： 通过训练模型，已经可以使用它来预测某些图像

predictions = model.predict(test_images)

# 以图像的方式查看完整的10类预测
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
               100*np.max(predictions_array),
               class_names[true_label]),
               color=color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()
    





