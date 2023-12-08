import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.preprocessing.image import ImageDataGenerator

model = load_model("Alzheimer/Resources/models/modeloVGG19.h5")

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

test_set = dataGenerator.flow_from_directory('Alzheimer/Resources/data/val/',
                                             target_size=(256, 256),
                                             batch_size=77,
                                             class_mode='categorical',
                                             shuffle=False)  

y_true = test_set.classes
y_pred_probs = model.predict(test_set)
y_pred = np.argmax(y_pred_probs, axis=1)

class_labels = list(test_set.class_indices.keys())

conf_matrix = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:")
print(report)
