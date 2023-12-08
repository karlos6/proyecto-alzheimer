import os
import cv2
from tensorflow.python.keras.models import load_model
from prediccion import prediccion
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras

# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from keras_preprocessing.image import load_img

class CustomAdam(tf.keras.optimizers.Adam):
    def __init__(self, *args, **kwargs):
        super(CustomAdam, self).__init__(*args, **kwargs)



model_builder = keras.applications.xception.Xception
img_size = (256, 266)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

clases= ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
last_conv_layer_name = "capa_1"

# Imagen de un paciente con demencia leve
#imagenRuta="C:/Users/carlo/Desktop/inteligentes/proyecto-alzheimer/Alzheimer/Resources/data/val/VeryMildDemented/31.jpg"

# Imagen de un paciente no demenciado
imagenRuta="C:/Users/carlo/Desktop/inteligentes/proyecto-alzheimer/Alzheimer/Resources/data/val/MildDemented/3.jpg"


img_path = keras.utils.get_file(
    "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
)

display(Image("dataset/test/3/0_2.jpg"))

# preparación de la imagen
def get_img_array(img_path, size):
    img = load_img(img_path,grayscale=True, target_size=size)
    array = img_to_array(img)
    print(array)
    array = np.expand_dims(array, axis=0)
    return array

# Configuración de gradiente ascendente
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Carga de modelo y predicción de imagen
img_array = preprocess_input(get_img_array(imagenRuta, (256,256)))
imagenes_aplanadas = img_array.reshape(-1, 65536)
model = tf.keras.models.load_model('Alzheimer/Resources/models/modeloCinco.h5', custom_objects={'CustomAdam': CustomAdam})
model.layers[-1].activation = None
miModeloCNN=prediccion("Alzheimer/Resources/models/modeloCinco.h5",256,256)
preds=miModeloCNN.predecir(imagen=cv2.imread(imagenRuta))
print("Predicted:", clases[preds])
print("La imagen cargada es ",clases[preds])
heatmap = make_gradcam_heatmap(imagenes_aplanadas, model, last_conv_layer_name)
plt.matshow(heatmap)
plt.show()

# Superposición de imagen y guardado
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = load_img(img_path)
    img = img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superposición de imagen
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Guardar imagen
    superimposed_img.save(cam_path)

    #  Mostrar imagen
    display(Image(cam_path))

# Guardar imagen
save_and_display_gradcam(imagenRuta, heatmap)


