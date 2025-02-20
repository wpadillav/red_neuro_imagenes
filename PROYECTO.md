## 🎯 **Proyecto: Red Neuronal para Clasificación de Imágenes con CNNs**

### 📌 **Objetivo del Proyecto**
Desarrollar un modelo de **visión por computadora** basado en **Redes Convolucionales (CNNs)** para **clasificar imágenes** y detectar objetos en ellas.

Ejemplos de aplicaciones:
- 📷 **Clasificación de imágenes**: Identificar si una imagen contiene un perro, gato, pájaro, etc.
- 🎯 **Detección de objetos**: Localizar y etiquetar objetos en imágenes (Ej: autos, peatones, señales de tránsito).
- 🏥 **Diagnóstico médico**: Identificar enfermedades en radiografías, tomografías, etc.

---

## 🔹 **1. Herramientas y Librerías**
Instalaremos las librerías necesarias en Python:

```bash
pip install tensorflow keras numpy matplotlib opencv-python pillow
```

- `tensorflow` y `keras`: Para construir la red neuronal CNN.
- `numpy`: Para manipulación de datos numéricos.
- `matplotlib`: Para visualizar imágenes y resultados.
- `opencv-python`: Para procesamiento de imágenes.
- `pillow (PIL)`: Para cargar imágenes.

---

## 🔹 **2. Obtención y Preparación de Datos**
Usaremos un dataset de imágenes. Puedes usar uno de **Kaggle** o **Google OpenImages**. Para este ejemplo, usaremos el dataset **"Dogs vs Cats"** de Kaggle:

### 📥 **Descargar el Dataset**
```bash
wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
unzip cats_and_dogs_filtered.zip
```

### 📂 **Estructura de las carpetas**
```
cats_and_dogs_filtered/
│── train/
│   ├── cats/  (Imágenes de gatos)
│   ├── dogs/  (Imágenes de perros)
│── validation/
│   ├── cats/
│   ├── dogs/
```

---

## 🔹 **3. Construcción del Modelo CNN**
Ahora diseñaremos una **Red Neuronal Convolucional (CNN)** con `Keras`:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Crear el modelo CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 1 salida (0 = gato, 1 = perro)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Mostrar estructura de la red
model.summary()
```

🔹 **Explicación:**
- **3 capas convolucionales** (`Conv2D`) para extraer características de la imagen.
- **Capas MaxPooling2D** para reducir la dimensionalidad.
- **Capa densa de 512 neuronas** para procesar las características extraídas.
- **Salida Sigmoide** (0 = gato, 1 = perro).

---

## 🔹 **4. Cargar y Preprocesar Imágenes**
Usaremos `ImageDataGenerator` para cargar las imágenes:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorios
train_dir = "cats_and_dogs_filtered/train"
validation_dir = "cats_and_dogs_filtered/validation"

# Preprocesamiento y aumento de datos
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Cargar imágenes
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
```

✅ **Explicación:**
- **Rescale=1./255** → Normaliza los píxeles a valores entre `0` y `1`.
- **Aumento de datos** (rotación, desplazamiento, zoom, etc.).
- **Cargar imágenes** con un tamaño de `150x150 px` y `batch_size=20`.

---

## 🔹 **5. Entrenamiento del Modelo**
Ahora entrenamos la CNN:

```python
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Número de batches por época
    epochs=20,            # Número de iteraciones sobre los datos
    validation_data=validation_generator,
    validation_steps=50
)
```

✅ **Explicación:**
- `epochs=20` → Se entrena por 20 ciclos completos.
- `steps_per_epoch=100` → Procesa 100 lotes por época.
- `validation_steps=50` → Evalúa en 50 lotes de validación.

---

## 🔹 **6. Evaluación del Modelo**
Después del entrenamiento, evaluamos su rendimiento:

```python
import matplotlib.pyplot as plt

# Gráfica de precisión y pérdida
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Precisión de entrenamiento')
plt.plot(epochs, val_acc, 'r', label='Precisión de validación')
plt.title('Precisión del Modelo')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'b', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, 'r', label='Pérdida de validación')
plt.title('Pérdida del Modelo')
plt.legend()
plt.show()
```

---

## 🔹 **7. Prueba del Modelo con una Imagen**
Probemos el modelo con una imagen nueva:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

# Cargar imagen
img_path = "prueba.jpg"  # Imagen de prueba
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión de batch

# Predicción
prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print("Es un perro 🐶")
else:
    print("Es un gato 🐱")
```

---

## 🚀 **8. Opcional: Guardar y Cargar el Modelo**
Si el modelo es bueno, podemos guardarlo:

```python
model.save("modelo_perros_gatos.h5")
```

Para cargarlo después:

```python
from tensorflow.keras.models import load_model
model = load_model("modelo_perros_gatos.h5")
```
