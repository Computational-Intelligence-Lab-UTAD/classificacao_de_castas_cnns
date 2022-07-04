# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:11:11 2021

Código utilizado para diminuir a quantidade de imagens do dataset. A ideia foi limitar em termos do tamanho do crop e da data.
@author: Gabriel
"""

import os
import datetime
import pandas as pd
import random
from shutil import copy2
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img


img_path = 'C:\\Users\\Gabriel\\OneDrive - Universidade de Tras-os-Montes e Alto Douro\\UTAD\\2020-2021\\Pesquisa\\Dataset\\Gerado\\dataset_versao_manual - ga'
#img_path = 'C:\\Users\\Gabriel\\Downloads\\imagens_geradas_menor_zoom800-800-v3'
teste_path = os.path.join(img_path, 'test')
validacao_path = os.path.join(img_path, 'validation')
treino_path = os.path.join(img_path, 'train')


outpath = 'C:\\Users\\Gabriel\\Downloads\\ga'

sep = os.path.sep
teste_path_out = os.path.join(outpath, 'test')
validacao_path_out = os.path.join(outpath, 'validation')
treino_path_out = os.path.join(outpath, 'train')


#verifica se as paths de saida existem
paths = [outpath, teste_path_out, validacao_path_out, treino_path_out]

for path in paths:
  if not os.path.lexists(path):
          os.mkdir(path)
  
#exemplo de nome de arquivo: 2017-05-09_1-0-0-13-300,600_600,900.jpg
def directory_from_df_with_class(df, directory):
    """
    Método responsável por transformar as imagens de um dataset num dataframe.
    Parameters
    ----------
    df : Dataframe pandas
        dataframe que será utilizado para povoar com os elementos. O dataframe será preenchido com os seguintes campos:
        ['path', 'class', 'date', 'zoom']
    directory : String
        diretório de onde as imagens serão retiradas.

    Returns
    -------
    df : dataframe pandas
        O dataframe povoado.
    """
    for r, d, f in os.walk(directory):
        for file in f:
          row = {}
          row['path'] = os.path.join(r, file)
          row['class'] = row['path'].split(os.path.sep)[-2]
          date_info =  file.split('_')[0].split('-')
          #captura a data, 
          row['date'] = datetime.datetime(int(date_info[0]), int(date_info[1]), int(date_info[2]))
          row['zoom'] = int(file.split('_')[1].split('-')[2])
          
          
          #print(row['class'])
          df= df.append(row, ignore_index=True)
    return df

#exemplo de nome de arquivo: 2017-05-09_1-0-0-13-300,600_600,900.jpg
def directory_from_df_with_class2(df, directory):
    """
    Método responsável por transformar as imagens de um dataset num dataframe.
    Parameters
    ----------
    df : Dataframe pandas
        dataframe que será utilizado para povoar com os elementos. O dataframe será preenchido com os seguintes campos:
        ['path', 'class', 'date', 'zoom']
    directory : String
        diretório de onde as imagens serão retiradas.

    Returns
    -------
    df : dataframe pandas
        O dataframe povoado.
    """
    for r, d, f in os.walk(directory):
        for file in f:
          row = {}
          row['path'] = os.path.join(r, file)
          row['class'] = row['path'].split(os.path.sep)[-2]
          
          
          #print(row['class'])
          df= df.append(row, ignore_index=True)
    return df

def rand_images_from_df(df, coluna_classe, quantidade=20, zooms = [0, 1, 2]):
    """
    Método responsável por sortear aliatoriamente uma quantidade de imagens definidas, por zoom, por data e por classe.
    Isso implica que a quantidade de imagens retornadas é: QUANTIDADE DE CLASSES * QUANTIDADE DATAS DE AQUISIÇÃO * QUANTIDADE DE ZOOMS * QUANTIDADE DE IMAGENS PASSADAS POR PARÂMETRO.

    Parameters
    ----------
    df : dataframe pandas
        Dataframe contendo as seguintes colunas: ['path', 'class', 'date', 'zoom'].
    coluna_classe : string
        Nome da coluna que representa a classe no dataset.
    quantidade : integer, optional
        DESCRIPTION. The default is 20. Quantidade de imagens que serão sorteadas, por zoom, por data e por tipo

    Returns
    -------
    df_result : dataframe pandas
        Dataframe contendo as seguintes colunas: ['path', 'class', 'date', 'zoom'] com as imagens sorteadas.

    """
    #captura as classes
    classes = df[coluna_classe].unique()
    columns_df = ['path', 'class', 'date', 'zoom']
    df_result = pd.DataFrame(columns=columns_df)
    
    
    #itera sobre as classes
    for classe in classes:
        df_classe = df[df[coluna_classe] == classe]
        #captura a as datas presentes para essa classe
        dates = df_classe['date'].unique()
        
        #itera sobre as datas
        for date in dates:
            df_date = df_classe[df_classe['date'] == date]
            #itera sobre os zooms
            for zoom in zooms:
                df_zoom = df_date[df_date['zoom'] == zoom]
                #se o dataset for menor que a quantidade, pula o sorteio adicionando todo ele ao dataset resultante
                if len(df_zoom) > quantidade:
                    #sorteia aleatoriamente
                    for i in range(quantidade):
                        indice = random.randint(0, len(df_zoom)-1)
                        df_result = df_result.append(df_zoom.iloc[[indice]])
                        #lista_imagens.append(df_classe.iloc[[indice]][path_classe].values[0])
                        df_zoom.drop(df_zoom.index[[indice]], inplace=True)
                else:
                    df_result = df_result.append(df_zoom, ignore_index=True)
    #retorna o dataset
    return df_result


def save_images_from_df(df, class_column, path_column, outpath):
    """
    Método responsável por copiar as imagens para o novo diretório

    Parameters
    ----------
    df : dataframe pandas
        Dataframe contendo as imagens que serão copiadas.
    class_column : string
        nome da coluna que representa a classe no dataframe.
    path_column : string
        nome da coluna que representa a path par a imagem no dataframe.
    outpath : string
        path para onde serão copiadas as novas imagens.

    Returns
    -------
    None.

    """
    for i, row in df.iterrows():
        if not row[path_column] is None:
            nome = row[path_column].split(os.path.sep)[-1]
            path_destino = os.path.join(outpath, row[class_column])
            if not os.path.lexists(path_destino):
                os.mkdir(path_destino)
            arquivo_path = os.path.join(path_destino, nome)
            copy2(row[path_column], arquivo_path)    

def augment_dataset(df, path_output, path_column='path', quantity=10):
    
    filenames=df[path_column].values
    for img in filenames:
        src_fname, ext = os.path.splitext(img) 
    
        datagen = ImageDataGenerator(rotation_range=50,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.6,1.0],
            vertical_flip=True)
    
    
        img = load_img(img)
    
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
    
        img_name = src_fname.split(os.path.sep)[-1]
        class_name = src_fname.split(os.path.sep)[-2]
        new_dir = os.path.join(path_output, class_name)
        if not os.path.lexists(new_dir):
            os.mkdir(new_dir)
            #save_fname = os.path.join(new_dir, os.path.basename(img_name))
    
        i = 0
        for batch in datagen.flow (x, batch_size=1, save_to_dir = new_dir, 
                           save_prefix = img_name, save_format='jpg'):
            i+=1
            if i>quantity:
                break

#criacao da dataframe
columns_df = ['path', 'class', 'date', 'zoom']
df = pd.DataFrame(columns=columns_df)

#faz dataset para treino
df_train = directory_from_df_with_class(df, treino_path)
print('finalizada a análise do diretório de treino')
#no conjunto de treino se captura 2.5x mais imagens
df_train_random = rand_images_from_df(df_train, 'class', quantidade=10, zooms = [1,2])
print('finalizado o sorteio do diretório de treino')

#faz dataset para validacao
df = pd.DataFrame(columns=columns_df)
df_val = directory_from_df_with_class(df, validacao_path)
print('finalizada a análise do diretório de validação')
df_val_random = rand_images_from_df(df_val, 'class', quantidade=5, zooms = [1,2])
print('finalizado o sorteio do diretório de validação')


#faz dataset para teste
df = pd.DataFrame(columns=columns_df)
df_test = directory_from_df_with_class(df, teste_path)
print('finalizada a análise do diretório de teste')
df_test_random = rand_images_from_df(df_test, 'class', quantidade=5, zooms = [1,2])
print('finalizado o sorteio do diretório de teste')

#salva as imagens
save_images_from_df(df=df_train_random, class_column='class', path_column='path', outpath=treino_path_out)
save_images_from_df(df=df_val_random, class_column='class', path_column='path', outpath=validacao_path_out)
save_images_from_df(df=df_test_random, class_column='class', path_column='path', outpath=teste_path_out)

df_train_random = directory_from_df_with_class2(df, treino_path)
augment_dataset(df_train_random, treino_path_out, quantity=10)


