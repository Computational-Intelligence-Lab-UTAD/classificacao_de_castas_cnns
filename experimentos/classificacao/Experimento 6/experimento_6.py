# -*- coding: utf-8 -*-

"""
Experimento 12, utilizar a estratégia de aplicar fine tunning ao xception para o mesmo dataset do experimento 6, 
só que dessa vez uti8lizando as bibliotecas de segmentação como uma etapa de preprocessamento 

Original file is located at
    https://colab.research.google.com/drive/1rabEiKMp50x4WCdWqo3QpHiZFc_euuPy

Baixando e extraindo o dataset:
"""
descricao_experimento = 'experimento12'
import os
from numpy.core.fromnumeric import size
import tensorflow as tf
from PIL import Image

"""Importando as bibliotecas necessárias."""

import tensorflow.keras as keras
import numpy as np 
from functools import partial, update_wrapper
import datetime
from tensorflow.keras import backend as K
from sklearn import metrics
import itertools
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras import models
import math

from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam,RMSprop, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

import cv2
from tensorflow.keras.applications.xception import preprocess_input

'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
'''

"""Trecho de código para salvar modelo no google drive:"""

diretorio = r'C:\Users\Gabriel\Downloads\segmentado_final'

nome_otmizador = '_stp_decay'
from requests import get
filename = F"{descricao_experimento} - " + diretorio.split(os.path.sep)[-1] 
print(filename)


tempo = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_save_name = filename+'_model_'+'_'+tempo+'.h5'
weight_save_name = filename+'_weight_'+'_'+tempo+'.h5'
path_model = F"C:\\Users\\Gabriel\\models\\{model_save_name}"
path_weight = F"C:\\Users\\Gabriel\\models\\{weight_save_name}"
path_model_ft = F"C:\\Users\\Gabriel\\models\\ft_{model_save_name}"
path_weight_ft = F"C:\\Users\\Gabriel\\models\\ft_{weight_save_name}"

"""obter resultados repreduzíveis independente de onde se execute"""



"""Código que calcula F1 score:"""

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

"""Método que visualiza os mapas de ativação das camadas convolucionais."""


def visualiza_imagens(model, imagens, target_size, layer_limiar=None, images_per_row=16, just_conv2d=True):
  #verifica se eh uma imagem ou uma lista de imagens
  if not isinstance(imagens, list):
    print('não reconheceu como lista')
    imagens = [imagens]
  
  #se limiar for nulo, deixa como sendo o maximo
  if layer_limiar == None:
    layer_limiar = len(model.layers) - 1

  layer_names = []
  for layer in model.layers[:layer_limiar]:
    layer_names.append(layer.name)

  #extrai a saida das primeiras layer_limar camadas
  layer_outputs = [layer.output for layer in model.layers[:layer_limiar]]
  #cria um modelo com a mesma entrada que o modelo original, entretando layer_limiar saidas, sendo cada saida dessas a saida de uma das layers definidas acima
  activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

  for path in imagens:
    #carrgea a imagem
    img = image.load_img(path, target_size=(target_size, target_size))
    #transforma em em array
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    #plota a imagem de teste
    plt.imshow(img_tensor[0])

    activations = activation_model.predict(img_tensor)

    for layer_name, layer_activation in zip(layer_names, activations):
      layer_name_token = layer_name.split('_')
      if (not just_conv2d) or (just_conv2d and (not layer_name_token[0] == 'batch') and (not layer_name_token[0] == 'dropout') and (not layer_name_token[0] == 'max_pooling2d')):
        #obtem o numero de features/canais        
        n_features = layer_activation.shape[-1]                               
        
        #obtem o tamanho do feature map
        size = layer_activation.shape[1]                                      

        #quantidade de colunas para plotar todos os canais do feature map
        n_cols = n_features // images_per_row

        #monta a grid                                 
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):                                             
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                :, :,
                                                col * images_per_row + row]
                #para se conseguir visualizar no espaco rgb
                channel_image -= channel_image.mean()                         
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,                   
                            row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

"""Código para se obter a matriz de confusão:"""

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def confusion_matrix(test_data_generator, model):
  test_data_generator.reset()
  predictions = model.predict_generator(test_data_generator, steps=test_set.samples)
  # Get most likely class
  predicted_classes = np.argmax(predictions, axis=1)
  true_classes = test_data_generator.classes
  class_labels = list(test_data_generator.class_indices.keys())
  print(class_labels)  

  report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
  cm = metrics.confusion_matrix(true_classes, predicted_classes)
  print(report)
  plot_confusion_matrix(cm, class_labels)

"""Método que plota história depois do treinamento e método que faz o teste:"""

import matplotlib.pyplot as plt
def plotar_historia(history, metrics=['acc', 'loss']):
  
  for metric in metrics:
    # summarize history for accuracy
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.title(metric.capitalize())
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

import numpy as np
def testar(test_set, modelo):
  test_set.reset()
  pred = modelo.predict_generator(test_set, steps=test_set.samples)
  predicted_class_indices = np.argmax(pred,axis=1)
  real_class_indices = test_set.labels

  
  labels=(test_set.class_indices)
  labels2=dict((v,k) for k,v in labels.items())

  predictions=[labels2[k] for k in predicted_class_indices]

  #print(predicted_class_indices)
  #print(np.array(real_class_indices))
  #print(np.array(predicted_class_indices))

  soma = np.sum(np.array(real_class_indices) == np.array(predicted_class_indices))
  #print(soma)
  accuracy_test = soma/test_set.samples
  
  return accuracy_test, real_class_indices, predicted_class_indices

"""Métodos para gerar LR:"""

#step decay
def step_decay(epoch):
   initial_lrate = 0.01
   flattern_factor = initial_lrate ** 2.25
   epochs_drop = 5.0
   #drop modelado como modelado no artigo
   drop = initial_lrate **(flattern_factor/epochs_drop)
   
   lrate = initial_lrate * math.pow(drop,  
           math.floor((epoch)/epochs_drop))
   return lrate

#step decay
def step_decay2(epoch):
   initial_lrate = 0.0001
   flattern_factor = initial_lrate ** 2.25
   epochs_drop = 5.0
   #drop modelado como modelado no artigo
   drop = initial_lrate **(flattern_factor/epochs_drop)
   
   lrate = initial_lrate * math.pow(drop,  
           math.floor((epoch)/epochs_drop))
   return lrate

""" Métodos para pre processamento """

seg_model_weights = r'C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\2020-2021\Pesquisa\Resultados\segmentação\Experimento 2\best_model.h5'
BACKBONE = 'efficientnetb3'
seg_target_size = (320, 320)
preprocess_input_segmentation = sm.get_preprocessing(BACKBONE)
seg_model = sm.Unet(BACKBONE, input_shape=(seg_target_size[0],seg_target_size[1],3), classes=1, activation='sigmoid')
seg_model.load_weights(seg_model_weights)
target_size_dimension = 300

def aply_mask2d_in_array(img, mask):
  copy_img = img.copy()
  

  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if mask[i, j] == 0:
          for k in range(3):
            copy_img[i, j, k] = 0
  
  return copy_img


def preprocessing(x):
    #create a copy of the image
    cop = np.copy(x)

    #resize the image to target size of the segmentation
    cop = cv2.resize(cop, seg_target_size)

    #expand dimension to put the image inside the model
    image = np.expand_dims(cop, axis=0)

    #preprocess the image before segmentation
    image = image/255.0

    #predict the mask
    pr_mask = seg_model.predict(image).round()

    #isolate the mask
    pr_mask = pr_mask[0,:,:,0]

    #resize the mask
    pr_mask = cv2.resize(pr_mask, (target_size_dimension, target_size_dimension))
    
    #apply the mask to the image, turning black the background part
    final = aply_mask2d_in_array(np.uint8(cv2.resize(x, (target_size_dimension, target_size_dimension))), pr_mask)
    
    #apply the method of preprocess of the specific backbone
    return preprocess_input(final)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))

"""Definição dos diretórios:"""

train_directory = os.path.join(diretorio, 'train')
val_directory = os.path.join(diretorio, 'validation')
test_directory = os.path.join(diretorio, 'test')

"""Divisão do dataset entre treino e teste, considerando o dataset não balanceado. A ideia aqui foi caputar a classe com menor quantidade de amostras e utilizar a mesma quantidade de amostras de teste para todas as redes neurais. """

#definição do data augmentation e geradores de treino, teste e validacao
batch = 16
target_size_dimension= 300
nclasses = 6


train_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

val_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_set = train_datagen.flow_from_directory(train_directory,
                                              target_size=(target_size_dimension, target_size_dimension),
                                              class_mode='categorical',
                                              batch_size=batch)

val_set = train_datagen.flow_from_directory(val_directory,
                                              target_size=(target_size_dimension, target_size_dimension),
                                              class_mode='categorical',
                                              batch_size=batch)

test_set = test_datagen.flow_from_directory(test_directory, 
                                            target_size=(target_size_dimension, target_size_dimension),
                                            class_mode='categorical',
                                            batch_size=1,
                                            shuffle=False)

"""Definição de entropia cruzada com os pesos:

Definição da focal loss:
"""

def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed




#strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#with strategy.scope():
with tf.device('/gpu:0'):
	model = keras.applications.Xception(
		include_top=False,
		input_shape= (target_size_dimension, target_size_dimension, 3),
		weights='imagenet',
		pooling='avg'
	)


	x = Dense(512, activation='relu')(model.output)
	x = Dropout(0.25)(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.25)(x)
	output = Dense(nclasses, activation='softmax')(x)
	nmodel = keras.models.Model(model.input, output)

	nmodel.compile(optimizer='sgd', loss='categorical_crossentropy',
			   metrics=['accuracy', f1_m,precision_m, recall_m])


#camadas que nao deverao ser treiandas, toda a parte convolucional da xception
for layer in range(len(model.layers)):
	model.layers[layer].treinable=False

loss_history = LossHistory()
#lrate = LearningRateScheduler(exp_decay)
estop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_m', patience=20)
lrate = LearningRateScheduler(step_decay)


model_checkpoint_weights_callback = keras.callbacks.ModelCheckpoint(
	filepath=path_weight,
	save_weights_only=True,
	monitor='val_f1_m',
	mode='max',
	save_best_only=True)

model_checkpoint_model_callback = keras.callbacks.ModelCheckpoint(
	filepath=path_model,
	save_weights_only=False,
	monitor='val_f1_m',
	mode='max',
	save_best_only=True)

callbacks_list = [loss_history, lrate, 
				model_checkpoint_weights_callback, 
				model_checkpoint_model_callback]


	
history = nmodel.fit_generator(train_set, steps_per_epoch=train_set.samples/batch,
			  epochs=100, validation_data=val_set, validation_steps=val_set.samples/batch,
			  shuffle=True, verbose=True, callbacks=callbacks_list)

#plotar_historia(history)
#faz loading do melhor modelo salvo no checkpoint, que eh o modelo com melhor resultado na etapa de treinamento anterior
nmodel.load_weights(path_weight)
print('A acurácia do teste do modelo básico é: ', testar(test_set, nmodel))

with tf.device('/gpu:0'):
	#learning rate diferente para a etapa de finetuning
	lrate2 = LearningRateScheduler(step_decay2)
	model_checkpoint_weights_callback = keras.callbacks.ModelCheckpoint(
		filepath=path_weight_ft,
		save_weights_only=True,
		monitor='val_f1_m',
		mode='max',
		save_best_only=True)

	model_checkpoint_model_callback = keras.callbacks.ModelCheckpoint(
		filepath=path_model_ft,
		save_weights_only=False,
		monitor='val_f1_m',
		mode='max',
		save_best_only=True)

	callbacks_list = [loss_history, estop, lrate2, model_checkpoint_weights_callback, model_checkpoint_model_callback]

	#etapas utilizadas para o fine tuning
	for layer in range(len(model.layers)):
		model.layers[layer].treinable=True

	nmodel.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', f1_m,precision_m, recall_m])
history = nmodel.fit(train_set, steps_per_epoch=train_set.samples/batch,
			  epochs=100, validation_data=val_set, validation_steps=val_set.samples/batch, shuffle=True, verbose=True, callbacks=callbacks_list)

nmodel.load_weights(path_weight_ft)


plotar_historia(history, metrics=['accuracy', 'loss', 'f1_m'])
accuracy, real_class_indices, predicted_class_indices = testar(test_set, nmodel)
print('A acurácia do teste do modelo básico é: ', accuracy)

confusion_matrix(test_set, nmodel)


