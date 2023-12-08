# Inportacion de librerías
import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
import cargaData
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19,preprocess_input

ancho=256
alto=256
numeroCanales=3
pixeles=ancho*alto


IMAGE_SIZE = [ancho, alto,numeroCanales]
formaImagen=(ancho,alto,numeroCanales)
nombreCategorias= ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

cantidaDatosEntrenamiento=[7168,5170,7679,7168]
cantidaDatosPruebas=[8959,6463,9599,8959]

model = VGG19( include_top = False,
            input_shape = IMAGE_SIZE,
            weights = 'imagenet')

model.summary()

for  layer in model.layers:
    layer.trainable = False

x = Flatten()(model.output)

prediction = Dense( len(nombreCategorias) , activation = 'softmax' )(x)

model = Model( inputs = model.input , outputs = prediction )

model.summary()

adam=Adam()


model.compile( loss = 'categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'] )


dataGenerator = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True ,
    fill_mode = 'nearest'
)


train_set = dataGenerator.flow_from_directory('Alzheimer/Resources/data/train/',
                                            target_size = ( 256 , 256 ),
                                            batch_size = 77,
                                            class_mode = 'categorical')


test_set = dataGenerator.flow_from_directory('Alzheimer/Resources/data/val/',
                                             target_size = ( 256 , 256 ),
                                            batch_size = 77,
                                            class_mode = 'categorical')

model.fit(  train_set,
            validation_data = test_set,
            epochs = 20,
            steps_per_epoch = 15,
            validation_steps = 30,
            verbose = 2)

ruta = "Alzheimer/Resources/models/modeloVGG19.h5"
model.save(ruta)
model.summary()

