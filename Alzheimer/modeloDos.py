# Importación de bibliotecas necesarias
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import InputLayer, Input, Conv2D, MaxPool2D, Reshape, Dense, Flatten
import cargaData  # Módulo personalizado para cargar datos

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Definición de parámetros para las imágenes
ancho = 256
alto = 256
pixeles = ancho * alto
numeroCanales = 1
formaImagen = (ancho, alto, numeroCanales)
nombreCategorias = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Definición de cantidades de datos para entrenamiento y pruebas
cantidaDatosEntrenamiento = [7168, 5171, 7680, 7168]
cantidaDatosPruebas = [8960, 6464, 9600, 8960]

# Carga de datos de entrenamiento
imagenes, probabilidades = cargaData.cargar_datos("Alzheimer/Resources/data/train/", nombreCategorias,
                                                 cantidaDatosEntrenamiento, ancho, alto)

# Creación del modelo de red neuronal convolucional (CNN) utilizando Keras
model = Sequential()
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))
model.add(Conv2D(kernel_size=2, strides=2, filters=40, padding="same", activation="relu", name="capa_1"))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(kernel_size=2, strides=2, filters=50, padding="same", activation="relu", name="capa_2"))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(len(nombreCategorias), activation="softmax"))

# Compilación del modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento del modelo
model.fit(x=imagenes, y=probabilidades, epochs=60, batch_size=60)

# Carga de datos de prueba
imagenesPrueba, probabilidadesPrueba = cargaData.cargar_datos_pruebas("Alzheimer/Resources/data/val/",
                                                                       nombreCategorias,
                                                                       cantidaDatosPruebas, cantidaDatosEntrenamiento,
                                                                       ancho, alto)

# Evaluación del modelo en datos de prueba
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("METRIC NAMES", model.metrics_names)
print("RESULTADOS", resultados)

# Guardado del modelo entrenado
ruta = "Alzheimer/Resources/models/modeloDos.h5"
model.save(ruta)

# Resumen del modelo
model.summary()

# Evaluación de métricas en datos de entrenamiento
metricResult = model.evaluate(x=imagenes, y=probabilidades)

# Predicciones en datos de prueba y creación de la matriz de confusión
scnn_pred = model.predict(imagenesPrueba, batch_size=60, verbose=1)
scnn_predicted = np.argmax(scnn_pred, axis=1)

# Creación de la matriz de confusión
scnn_cm = confusion_matrix(np.argmax(probabilidadesPrueba, axis=1), scnn_predicted)

# Visualización de la matriz de confusión
scnn_df_cm = pd.DataFrame(scnn_cm, range(4), range(4))
plt.figure(figsize=(20, 14))
sn.set(font_scale=1.4)  # para el tamaño de la etiqueta
sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12})  # tamaño de la fuente
plt.show()

# Informe de clasificación
scnn_report = classification_report(np.argmax(probabilidadesPrueba, axis=1), scnn_predicted)
print("SCNN REPORT", scnn_report)
