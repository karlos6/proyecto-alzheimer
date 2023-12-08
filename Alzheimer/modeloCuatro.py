from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
import cargaData
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ancho=256
alto=256
pixeles=ancho*alto
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
nombreCategorias= ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

cantidaDatosEntrenamiento=[7168,5171,7680,7168]
cantidaDatosPruebas=[8960,6464,9600,8960]

imagenes, probabilidades= cargaData.cargar_datos("Alzheimer/Resources/data/train/",nombreCategorias,cantidaDatosEntrenamiento,ancho,alto)

model=Sequential() 
model.add(InputLayer(input_shape=(pixeles,))) 


model.add(Reshape(formaImagen))

model.add(Conv2D(kernel_size=4,strides=2,filters=32,padding="same",activation="relu",name="capa_1")) 

model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=4,strides=2,filters=64,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=4,strides=2,filters=128,padding="same",activation="relu",name="capa_3"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Flatten()) 
model.add(Dense(128,activation="relu")) # Agregar capa densa de 128 pixxeles

model.add(Dense(len(nombreCategorias),activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",  metrics=["accuracy"])

model.fit(x=imagenes,y=probabilidades,epochs=32,batch_size=60)

imagenesPrueba,probabilidadesPrueba= cargaData.cargar_datos_pruebas("Alzheimer/Resources/data/val/",nombreCategorias,cantidaDatosPruebas, cantidaDatosEntrenamiento,ancho,alto)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("RESULTADOS", resultados)

ruta="Alzheimer/Resources/models/modeloCuatro.h5"
model.save(ruta)
model.summary()

metricResult = model.evaluate(x=imagenes, y=probabilidades)

scnn_pred = model.predict(imagenesPrueba, batch_size=60, verbose=1)
scnn_predicted = np.argmax(scnn_pred, axis=1)

# Creamos la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(probabilidadesPrueba, axis=1), scnn_predicted)

# Visualiamos la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(4), range(4))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

scnn_report = classification_report(np.argmax(probabilidadesPrueba, axis=1), scnn_predicted)
print("SCNN REPORT", scnn_report)