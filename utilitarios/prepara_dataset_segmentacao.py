# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:24:25 2021

Código utilizado para gerar as máscaras e as imagens a partir dos videos e marcações XML geradas pelo sensarea.

@author: Gabriel
"""

import cv2 
import glob
import os
import pandas as pd
import datetime
import shapely
import xml.etree.ElementTree as et
from shapely.geometry import Polygon
#import geopandas as gpd
from shapely.ops import unary_union
import numpy as np
import cv2




def get_polygons(path):
    xtree = et.parse(path)
    xroot = xtree.getroot()
    
    masks = xroot.find("masks")
    df = pd.DataFrame(columns=['frame', 'polygon']) 
    
    for mask in masks:
        
        dic = {}
        dic['frame'] = int(mask.find('frame').text)
        pontos = mask.find('polygon').attrib.get('points')
        
        tokens = pontos.split(' ')
        poly = []
        for token in tokens[:-1]:
            ponto = token.split(',')
            poly.append((int(ponto[0]), int(ponto[1])))
        
        dic['polygon'] = poly
        df = df.append(dic, ignore_index=True)
    
    return df


#fonte: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

path = "C:\\Users\\Gabriel\\OneDrive - Universidade de Tras-os-Montes e Alto Douro\\UTAD\\2020-2021\\Pesquisa\\Dataset\\Vídeos\\organizado\\dividido"
path_xml = "C:\\Users\\Gabriel\\OneDrive - Universidade de Tras-os-Montes e Alto Douro\\UTAD\\2020-2021\\Pesquisa\\Dataset\\Vídeos\\mascaras"
os.chdir(path)

#image_quantity=20



#configuracoes de tamanho: [tamanho da imagem, tamanho do stride, percentual minimo de planta para salvar a imagem]
configs = [[512, int(512/2), 0.80, 0.5], [800, int(800/3), 0.80, 0.5]]

#quantidade de frames que serão "pulados"
next_frame = 30

#dimensoes para salvar
dim_salvar = 512
tam_salvar = (dim_salvar, dim_salvar)

columns = ['path', 'class', 'name', 'data_str', 'dia', 'mes', 'ano', 'set']
df=pd.DataFrame(columns=columns)
parser = {'junho':'06',
          'mai':'05',
          'agosto':'08',
          'julho':'07',
          'set':'09',
          'maio':'05',
          'ago':'08'}

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".MTS"):
             dic = {}
             dic['path'] = os.path.join(root, file)
             dic['name'] = dic['path'].split(os.path.sep)[-1]
             dic['class'] = dic['path'].split(os.path.sep)[-2]
             dic['set'] = dic['path'].split(os.path.sep)[-3]
             df=df.append(dic, ignore_index=True)

print(df.head())

#verifiva se diritorio existe
path_to_save = 'C:\\Users\\Gabriel\\Downloads\\teste_masks_crop3'
if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)


for i, row in df.iterrows():
    
    for folder in ['image', 'mask']:
        #verificar se diretorio para salvar imagem existe: diretorio/conjunto/classe
        if not os.path.exists(os.path.join(path_to_save, row['set'], folder)):
            #os.mkdir(os.path.join(path_to_save, folder, row['set']))
            os.makedirs(os.path.join(path_to_save, row['set'], folder), exist_ok=True)
                              
        #if not os.path.exists(os.path.join(path_to_save, folder, row['set'],row['class'])):
            #os.mkdir(os.path.join(path_to_save, folder, row['set'], row['class']))
            #os.makedirs(os.path.join(path_to_save, folder, row['set'], row['class']), exist_ok=True)
    
    #le video
    cap = cv2.VideoCapture(row['path'])
    
    #captura quantidade de frames do video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #define path e le arquivo de poligonos
    path_polygon = os.path.join(path_xml, row['class'], row['name'].split('.')[0]+'.xml')
    
    #verifica se o arquivo existe
    df_polygon = None
    if os.path.isfile(path_polygon):
        df_polygon = get_polygons(os.path.join(path_xml, row['class'], row['name'].split('.')[0]+'.xml'))
    n_frame = 0
    continua = True
    print('Video: ', row['path'])
    while(continua):
        #cap.set(cv2.CAP_PROP_POS_FRAMES, (step*i)-1)
        res, frame = cap.read()
        
        if res:
            try:
                #captura altura e largura do frame
                altura_imagem, largura_imagem = frame.shape[:2]
                
                #como o indice e igual ao numero do frame pode-se usar a busca direta no dataframe para se obter o poligono
                #e entao gerar uma representacao com a biblioteca shapely
                frame_polygon = None
                #verifica se arquivo existe
                if not df_polygon is None:
                    frame_polygon = Polygon(df_polygon.iloc[n_frame]['polygon']).buffer(0)
                    
                    #transformar multipoligono num poligono so
                    if frame_polygon.geom_type == 'MultiPolygon': 
                        
                        frame_polygon =unary_union(list(frame_polygon))
                        if frame_polygon.geom_type == 'MultiPolygon':
                        
                            polys = list(frame_polygon)
                            area = 0
                            index = None
                            
                            for i, pl in enumerate(polys):
                                if pl.area > area:
                                    area = pl.area
                                    index = i
                            
                            frame_polygon = polys[index]
                           
                            
                        
                    #mask = np.zeros((altura_imagem, largura_imagem))
                    mask = np.full((altura_imagem, largura_imagem), 0, dtype=np.uint8)
                    
                    
                    
                    coords = np.array([[x[0], x[1]] for x in list(frame_polygon.exterior.coords)], dtype=np.int32)
                    
                    coords = coords.reshape((-1,1,2))
                    
                    mask = cv2.fillPoly(mask,[coords], 1)
                
                    #itera sobre as configuracoes de recorte
                    for c, config in enumerate(configs):
                        #definicao de tamanho de recorte para os quadrados
                        image_size_side = config[0]
                        #para definir o overlaping entre as imagens, nesse caso sem overlaping
                        stride = config[1]
                        
                        #percentual minimo de planta necessario no poligono para salvar 
                        percentual_maximo = config[2]
                        percentual_minimo = config[3]
                        altura = 0
                        #itera sobre a altura da imagem
                        k = 0
                        
                        
                        while(altura+image_size_side < altura_imagem):
                            
                            largura = 0
                            #itera sobre a largura da imagem
                            while(largura+image_size_side < largura_imagem):
                                #corta imagem
                                crop_img = frame[altura:altura+image_size_side, largura:largura+image_size_side]
                                
                                #corta poligono
                                crop_polygon = Polygon([(altura, largura), (altura, largura+image_size_side), (altura+image_size_side, largura+image_size_side), (altura+image_size_side, largura)])
                                
                                #corta mascara
                                crop_mask = mask[altura:altura+image_size_side, largura:largura+image_size_side]
                                
                                #resize das imagens
                                resized = cv2.resize(crop_img, tam_salvar, interpolation = cv2.INTER_AREA)
                                resized_mask = cv2.resize(crop_mask, tam_salvar, interpolation = cv2.INTER_AREA)
                                
                                #se o poligono existir faz os calculos para verificar se salva os crops
                                if (crop_polygon.intersects(frame_polygon)):
                                    area_interseccao = crop_polygon.intersection(frame_polygon).area
                                    area_crop_polygon = crop_polygon.area
                                    
                                    if(area_interseccao/area_crop_polygon > percentual_minimo and area_interseccao/area_crop_polygon <= percentual_maximo):
                                        cv2.imwrite(os.path.join(path_to_save, row['set'], 'image', row['class']+'-'+ row['name'].split('.')[0]+'-{}-{}-{}-{},{}_{},{}.jpg'.format(n_frame, c, k, largura, altura, largura+image_size_side, altura+image_size_side)), resized)
                                        cv2.imwrite(os.path.join(path_to_save, row['set'], 'mask', row['class']+'-'+ row['name'].split('.')[0]+'-{}-{}-{}-{},{}_{},{}.png'.format(n_frame, c, k, largura, altura, largura+image_size_side, altura+image_size_side)), resized_mask)
                                        # print('nome: ', os.path.join(path_to_save, row['set'], row['class'], row['name']+'-{}-{}.jpg'.format(n_frame, k)))
                                        # print('area interseccao: ', area_interseccao)
                                        # print('area crop: ', area_crop_polygon)
                                        # print('percentual de area: ', area_interseccao/area_crop_polygon)
                                        # print('blurry: ', variance_of_laplacian(crop_img))
                                        # print('---------------------------------\n')
                                        
                                
                                
                                #atualiza a largura com o valor do passo
                                largura = largura + stride
                                k = k+1
                            #atualiza a altura com o valor do passo
                            altura = altura+stride
                #caso o resultado nao exista significa que nao há mais frames e portanto e hora de parar
            except:
                print("Something else went wrong")
        else:
            continua = False
            print('read falso')
            
        #avanca frames manualmente em 15 frames
        for i in range(next_frame):
            cap.read()
        
        #conta o avanco mais um, porque no proximo read o avanco nao e contabilizado...
        n_frame = n_frame + next_frame + 1
        
        #se avanco se estender aos últimos 10 frames ignora
        if n_frame > length-10:
            continua = False
            #print('ignorando os 10 ultimos frames para evitar perda de processamento e imagens borradas nos videos que nao foram marcados')
        #print('=======================================\n')
    
    #libera o video
    cap.release()