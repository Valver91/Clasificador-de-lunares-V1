# 🧠 Clasificador de Lunares Malignos usando Deep Learning y Criterios ABCDE

Este proyecto es una prueba de concepto para detectar lunares malignos a partir de imágenes de la piel, combinando visión por computadora con criterios clínicos. Actualmente se basa en un modelo de clasificación binaria entrenado con imágenes categorizadas como **benignas** o **malignas**, y está evolucionando hacia una aproximación más basada en características clínicas cuantificables (criterios ABCDE del melanoma).

---

## 🔍 Objetivo

Desarrollar un sistema que ayude a detectar de forma temprana indicios de melanoma a partir de imágenes, combinando:

- **Deep Learning** sobre imágenes.
- **Extracción automática de características clínicas** (asimetría, borde, color, diámetro).
- Posible integración de modelos híbridos para mejorar la precisión con datasets pequeños.

---
## 🧪 Versión Actual del Proyecto

### ✅ Entrenamiento con imágenes clasificadas

- Utiliza `ImageDataGenerator` de Keras para cargar y procesar imágenes.
- Entrenamiento de una red neuronal convolucional (CNN) básica.
- Clasificación binaria: **benigno (0)** vs **maligno (1)**.
- Resultados iniciales aceptables, pero aún por debajo de la precisión esperada.

> **Limitación:** Dataset pequeño y potencial overfitting por usar data augmentation agresivo.

---

## 🔄 Evolución del Proyecto

Actualmente estoy trabajando en una segunda versión que busca mejorar el rendimiento del modelo utilizando **información estructurada extraída de las imágenes**:

### 🧭 Enfoque basado en ABCDE del melanoma:

- **A**simetría
- **B**orde irregular
- **C**olor no uniforme
- **D**iámetro
- (Más adelante podría incluirse la **E**volución temporal)

### 🔧 Plan de trabajo:

1. Convertir imágenes en un CSV con etiquetas clínicas.
2. Crear funciones de análisis que calculen medidas como:
   - Índice de asimetría.
   - Irregularidad del borde.
   - Diversidad de colores.
   - Tamaño relativo.
3. Entrenar un modelo (ML clásico o red neuronal) usando ese CSV como entrada.
4. Comparar resultados con el modelo basado exclusivamente en imágenes.

---

## 📊 Resultados esperados

Con esta evolución se espera:

- Mayor **precisión** en la clasificación.
- Menor riesgo de **overfitting**.
- Posibilidad de explicar decisiones del modelo (modelo más interpretable).

---

## 🚀 Tecnologías Utilizadas

- Python 3.10+
- TensorFlow / Keras
- NumPy / Pandas
- OpenCV (para análisis de imagen clínica, en desarrollo)

---

## 🧠 Autor

**Santiago Fernández**  
Estudiante de Ingeniería Biomédica | Apasionado por el ML y la visión artificial  
[LinkedIn](https://www.linkedin.com/in/santiago-fern%C3%A1ndez-351909247/) | [Blog Técnico](https://santiagofv.com/)

---

## ⚠️ Disclaimer

Este proyecto es puramente educativo y **no debe utilizarse como herramienta de diagnóstico médico**. El objetivo es explorar técnicas de Machine Learning aplicadas a imágenes médicas.


