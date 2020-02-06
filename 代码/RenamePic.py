# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:45:54 2020

@author: djasl
"""

import os  
 
def rename(file_dir,name):
    '''将网上爬下来的图片重命名（更好的观看）'''
    '''第一个参数是目标文件名 第二个参数是图片的名称'''
    i=0
    for file in os.listdir(file_dir):  
        '''获取该路径文件下的所有图片'''    
        src = os.path.join(os.path.abspath(file_dir), file) 
        '''修改后图片的存储位置（目标文件夹+新的图片的名称）'''        
        dst = os.path.join(os.path.abspath(file_dir),  name+str(i) + '.jpg')
        os.rename(src, dst) #将图片重新命名
        i=i+1     

def getdir(file_dir):
    for root, dirs, files in os.walk(file_dir):
        array = dirs
        if array:
            return array
    
        
#
#file_dir="./dataset/images/"  #目标下的文件夹名称
#
#dirs = getdir(file_dir)  #获取目标路径下的文件夹名称
#
#for dir_name in dirs:
#    rename(file_dir+'/'+dir_name,dir_name)  #获取目标文件夹下，子文件的的路径 并进行重命名


