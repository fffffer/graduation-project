# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:12:02 2020

@author: djasl
"""
# 图像预处理
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 调整亮度.对比度.饱和度.色相的顺序可以得到不同的结果
# 预处理时随机选择的一种,降低无关因素对模型的影响
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering ==1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering ==2:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering ==3:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering ==4:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering ==5:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)

# 给定解码后的图像.目标图像的尺寸以及图像上的标注框
def preprocess(image, height, width, bbox):
    # 若没有提供标注框则默认为关注区域为整个图像
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # 转换图像数据类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # 随机截取图像减小识别物体大小对模型的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 随机调整图像的大小
    distorted_image = tf.image.resize_images(distorted_image, (height, width), method=np.random.randint(4))
    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


source_file="./dataset/images/Bag/"       #原始文件地址
target_file="./dataset/images_enhance/Bag/"       #目标文件地址
file_list=os.listdir(source_file)   #读取原始文件的路径

with tf.Session() as sess:
    index = 0
    for filename in file_list:
        # 获取图像
        image_raw_data = tf.gfile.FastGFile(source_file+filename, 'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data)
        boxes = tf.constant([[[0.1, 0.32, 0.8, 0.7]]])
        # 获得9种不同的图像并显示结果
        for i in range(4):
            # 将图像大小调整为200*200
            result = preprocess(img_data, 200, 200, boxes)
            
            image_data=tf.image.convert_image_dtype(result,dtype=tf.uint8)
     
            encode_data=tf.image.encode_jpeg(image_data)
            with tf.gfile.GFile(target_file+str(index)+"_enhance"+".jpeg","wb") as f:
                index = index + 1
                f.write(encode_data.eval())
            
print("数据增强完成！")