import os
from django.shortcuts import render
from .forms import ImageUploadForm

# Manejo de imagenes
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Libreria para el deep learning
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model/mnist_model.h5')
model = load_model(MODEL_PATH)


# Función para preprocesar la imagen
def preprocess_image(image):
    #TODO Convertir la imagen a escala de grises y redimensionar
    img = image.convert('L')
    img = img.resize((28, 28)) # redimensionar la imagen
    img_array = np.array(img)
    img_array = 255 - img_array  # Invertir colores
    img_array = img_array / 255.0  # Normalizar

    # Expandir dimensiones para que sea compatible con el modelo
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# vista principal
def index(request):
    # variables que se usaran como parametros para la vista
    prediction = None
    error = None
    img_base64 = None
    form = ImageUploadForm()

    # Si hay un reques del formulario
    if request.method == 'POST':
        # Si el formulario es de carga de imagen
        if 'upload_image' in request.POST:
            # Iniciarlizar el formulario
            form = ImageUploadForm(request.POST, request.FILES)

            # Validar el formulario
            if form.is_valid() and 'image' in request.FILES:
                # Obtener la imagen del formulario
                image = form.cleaned_data['image']

                # Preprocesar la imagen
                img = Image.open(image)
                img_array = preprocess_image(img)

                # Realizar la predicción
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions)
                prediction = f'El número predicho es: {predicted_class}'

                # Convertir la imagen a base64 para mostrarla como vista previa
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            else:
                error = "Debe incluir una imagen válida para hacer la predicción."

    return render(request, 'classifier/index.html', {
        "form": form,
        "error": error,
        "prediction": prediction,
        "uploaded_image": img_base64,
    })



