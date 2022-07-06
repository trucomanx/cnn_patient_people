#!/usr/bin/python

import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import cv2
import numpy as np

def load_model_patient_people(file_of_weight=''):
    '''
    Retorna un modelo para la clasificación de {patient, people}.
    Adicionalmente, si el archivo `file_of_weight` existe los pesos son cargados.
    
    :param file_of_weight: Archivo donde se encuentran los pesos.
    :type file_of_weight: str
    :return: Retorna un modelo de red neuronal
    :rtype: tensorflow.python.keras.engine.sequential.Sequential
    '''
    # una capa cualquiera en tf2
    url='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
    mobilenetv2 = hub.KerasLayer(url,input_shape=(224,224,3))
    mobilenetv2.trainable =False

    # modelo nuevo
    modelo = tf.keras.Sequential([
        mobilenetv2,
        tf.keras.layers.Dense(2,activation='softmax')
    ])
    
    if os.path.exists(file_of_weight):
        modelo.load_weights(file_of_weight)

    return modelo

def evaluate_model_from_file(modelo, imgfilepath):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde el archivo `imgfilepath`.
    
    :param modelo: Modelo de la red neuronal.
    :type modelo: tensorflow.python.keras.engine.sequential.Sequential
    :param imgfilepath: Archivo de donde se leerá la imagen a testar.
    :type imgfilepath: str
    :return: Retorna la classificación, True para patient y False para people.
    :rtype: bool
    '''
    image = load_img(imgfilepath)
    image = img_to_array(image)
    image=cv2.resize(image,(224,224));
    image = np.expand_dims(image, axis=0)
    image=image/255.0;
    res=modelo.predict(image.reshape(-1,224,224,3));
    
    return res[0][0]>res[0][1];

def evaluate_model_from_pil(modelo, image):
    '''
    Evalua la red neuronal descrita en `modelo`, la entrada es leida desde una imagen PIL.
    
    :param modelo: Modelo de la red neuronal.
    :type modelo: tensorflow.python.keras.engine.sequential.Sequential
    :param image: Imagen a testar.
    :type image: PIL.PngImagePlugin.PngImageFile
    :return: Retorna la classificación, True para patient y False para people.
    :rtype: bool
    '''
    image=np.array(image)
    image=cv2.resize(image,(224,224));
    image = np.expand_dims(image, axis=0)
    image=image/255.0;
    res=modelo.predict(image.reshape(-1,224,224,3));
    
    return res[0][0]>res[0][1];