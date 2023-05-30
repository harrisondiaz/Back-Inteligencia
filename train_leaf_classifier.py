import os
import cv2
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback


excel_file = 'Datos Imágenes (1).xlsx'
df = pd.read_excel(excel_file)

def download_and_process_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img = img.resize((128, 128))
        return np.array(img)
    except:
        return None


imagenes = []
etiquetas = []
for i, row in df.iterrows():
    img = download_and_process_image(row['URL repositorio'])
    if img is not None:
        imagenes.append(img)
        etiquetas.append(row['Fenologia'])
    print(f"Procesando imagen {i + 1} de {len(df)}")

imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)


X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)


X_train = X_train / 255.0
X_test = X_test / 255.0

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(etiquetas)), activation='softmax')
])


modelo.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


print_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(
        f"Época {epoch + 1}: Precisión entrenamiento = {logs['accuracy']}, Precisión validación = {logs['val_accuracy']}"))

modelo.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[print_callback])

modelo.save('modelo_entrenado_excel.h5')
np.save('encoder_classes_excel.npy', encoder.classes_)