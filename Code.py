# -*- coding: utf-8 -*-
"""
@author: Santiango Fernández

Prueba de detección de lunares malignos.
"""

# Importación de las librerias necesarias.
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Rutas a las carpetas de datos
data_dir = Path(r"#")
benign_dir = data_dir / "benign"
malignant_dir = data_dir / "malignant"

# Parámetros de configuración
img_height, img_width = 224, 244
batch_size = 40

# trabajar sobre los ficheros organizadas por carpetas (ejemplo)
# Crear un generador de datos con augmentación
# ojo con las deformaciones porque puede ser contra.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Usaremos el 20% de los datos para validación
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Crear generadores para entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Subconjunto de entrenamiento
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Subconjunto de validación
)


#Crear y compilar el modelo
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.7),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# Definir callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)


# Evaluar el modelo en el conjunto de validación
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")


#Guardar el modelo entrenado
model.save('skin_detector.h5')




