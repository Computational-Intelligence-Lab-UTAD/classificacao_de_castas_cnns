# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:20:47 2021
Redimensiona imagens de uma pasta
@author: Gabriel
"""

import os 
from pathlib import Path
import cv2

dir_to_save = r'C:\Users\Gabriel\Downloads\final update'
tam = (512,512)
date = '22 jul'

# Print png images in folder
for i, fpath in enumerate(Path(r'C:\Users\Gabriel\Downloads\castas v3-22 jul-tablet').rglob('*.jpg')):
    filepath= str(fpath)
    #verifica se pasta existe
    path = os.path.join(dir_to_save, filepath.split(os.path.sep)[-2])
    os.makedirs(path, exist_ok = True)
    
    image = cv2.imread(filepath)
    
    resized = cv2.resize(image, tam, interpolation = cv2.INTER_AREA)
    
    
    image_name = os.path.join(path, filepath.split(os.path.sep)[-3].split("-")[-1]+'-'+date+'-'+filepath.split(os.path.sep)[-2].split(".")[0]+'-'+str(i)+'.jpg')
    #image.save(os.path.join(path, filepath.split(os.path.sep)[-1].split(".")[0]+'.jpg'), "JPEG")
    print(image_name)
    cv2.imwrite(image_name, resized)