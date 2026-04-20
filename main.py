import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, EfficientNetB0, Xception
)
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load models
models = {
    "VGG16": VGG16(weights='imagenet'),
    "ResNet50": ResNet50(weights='imagenet'),
    "MobileNetV2": MobileNetV2(weights='imagenet'),
    "EfficientNetB0": EfficientNetB0(weights='imagenet'),
    "Xception": Xception(weights='imagenet')
}

# Load image
img_path = 'test3.jpg'

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess
img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)

# Predict
for name, model in models.items():
    print(f"\n{name} Predictions:")
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    
    for i, (id, label, prob) in enumerate(decoded):
        print(f"{i+1}: {label} ({prob:.2f})")