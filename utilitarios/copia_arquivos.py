# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:24:25 2021

Script utiliado para copiar e renomear os vídeos, de modo que o nome refletisse a data em que o mesmo foi gravado.
@author: Gabriel
"""

import cv2 
import glob
import os
import pandas as pd
import datetime
from shutil import copyfile


path = 'C:\\Users\\Gabriel\\Downloads\\FolhasCastasV1_Gabriel\\FolhasCastasV1\\UTAD'
os.chdir(path)

columns = ['path', 'class', 'name', 'data_str', 'dia', 'mes', 'ano']
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
             dic['class'] = dic['path'].split(os.path.sep)[-4]
             dic['name'] = dic['path'].split(os.path.sep)[-2]+'_'+dic['path'].split(os.path.sep)[-1].split('.')[0]
             dic['data_str'] =  dic['path'].split(os.path.sep)[-3]
             data_split = dic['data_str'].split('_')
             dia = data_split[-3]
             mes = data_split[-2].lower()
             ano = data_split[-1]
             dic['dia'] = dia
             dic['mes'] = parser[mes]
             dic['ano'] = ano
             dic['data'] = ano+'-'+parser[mes]+'-'+dia
             df=df.append(dic, ignore_index=True)
             


diretorio_para_copiar = 'C:\\Users\\Gabriel\\Downloads\\FolhasCastasV1_Gabriel\\FolhasCastasV1\\organizado'
with pd.ExcelWriter(diretorio_para_copiar+'/relatorio_desorganizado.xlsx') as writer:
    df.to_excel(writer, index=False)
    

if not os.path.exists(diretorio_para_copiar):
    os.mkdir(diretorio_para_copiar)
    
df_novo = pd.DataFrame(columns=['classe','data', 'arquivo'])
dicionario_controle_nomes = {}
for i, row in df.iterrows():
    class_dir = os.path.join(diretorio_para_copiar, row['class'])
   
    #verifica se diretorio da classe nao existe
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    
    if not row['class'] in dicionario_controle_nomes:
        dicionario_controle_nomes[row['class']] = {}
    
    arquivos_existentes = dicionario_controle_nomes[row['class']]
    
    numeracao = 0
    if row['data'] in arquivos_existentes:
        numeracao = arquivos_existentes[row['data']]
    arquivos_existentes[row['data']] = numeracao + 1
    
    name_file = row['data']+'_'+str(numeracao)+'.MTS'
    path_file = os.path.join(class_dir, name_file)
    
    copyfile(row['path'], path_file)
    
    dicionario_arquivo = {'classe':row['class'],
                          'data':row['data'],
                          'arquivo':path_file}
    
    df_novo = df_novo.append(dicionario_arquivo, ignore_index=True)
    
with pd.ExcelWriter(diretorio_para_copiar+'/relatorio.xlsx') as writer:
    df_novo.to_excel(writer, index=False)
    