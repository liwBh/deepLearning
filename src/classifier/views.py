import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from django.shortcuts import render, get_object_or_404
from tensorflow.keras.models import load_model

from .forms import ImageUploadForm, CorrectionForm
from .models import Prediction

# Ruta donde se guardarán las imágenes
IMAGE_DIR = '/home/liwbh/Documentos/ProyectosU/deepLearning/src/classifier/static/classifier/img/numbers/'

# Cargar el modelo previamente entrenado
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model/mnist_model.h5')
model = load_model(MODEL_PATH)


# Función para preprocesar la imagen
def preprocess_image(image):
    # Convertir la imagen a escala de grises y redimensionar
    img = image.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # Invertir colores
    img_array = img_array / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para que sea compatible con el modelo

    return img_array


def index(request):
    prediction = None
    error = None
    prediction_obj = None
    form = ImageUploadForm()  # Asegurarte de que form esté definido
    form_correction = CorrectionForm()  # Asegurarte de que form_correction esté definido

    if request.method == 'POST':
        # Si el formulario es de carga de imagen
        if 'upload_image' in request.POST:
            form = ImageUploadForm(request.POST, request.FILES)  # Actualizar form aquí
            if form.is_valid() and 'image' in request.FILES:
                # Obtener la imagen del formulario
                image = form.cleaned_data['image']

                # Crear instancia
                img_instance = Prediction(image=image)

                # Preprocesar la imagen y hacer la predicción
                img = Image.open(image)
                img_array = preprocess_image(img)
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions)
                prediction = f'El número predicho es: {predicted_class}'

                # Guardar la predicción en la base de datos
                img_instance.prediction = predicted_class
                img_instance.save()

                # Objeto para usar en el formulario de corrección
                prediction_obj = img_instance
            else:
                error = "Debe incluir una imagen válida para hacer la predicción."

        # Si el formulario es de corrección
        elif 'correct_prediction' in request.POST:
            prediction_id = request.POST.get('prediction_id')
            form_correction = CorrectionForm(request.POST)  # Actualizar form_correction aquí

            if form_correction.is_valid():
                # Obtener la corrección, el número correcto
                correction = form_correction.cleaned_data['correction']
                # En caso de no encontrarlo
                prediction_obj = get_object_or_404(Prediction, id=prediction_id)

                # Actualizar la predicción con la corrección
                prediction_obj.correction = correction
                prediction_obj.fail = False
                prediction_obj.save()

                # Obtener los datos corregidos
                X_train, y_train = get_corrected_data()

                # Reentrenar el modelo
                model.fit(X_train, y_train, epochs=5, batch_size=32)
                # Guardar el modelo actualizado
                model.save(MODEL_PATH)

                prediction = f'La corrección ha sido aplicada. El número corregido es: {correction}.'

    return render(request, 'classifier/index.html', {
        "form": form,
        "form_correction": form_correction,
        "error": error,
        "prediction": prediction,
        "prediction_obj": prediction_obj
    })

def get_corrected_data():
    # Obtener las predicciones corregidas
    corrected_predictions = Prediction.objects.filter(fail=False).exclude(correction__isnull=True)

    X_train = []
    y_train = []

    for prediction in corrected_predictions:
        # Cargar la imagen
        img = Image.open(prediction.image.path)
        img_array = preprocess_image(img)  # Aplicar el mismo preprocesamiento que usas para las predicciones
        X_train.append(img_array)

        # Usar la corrección como la etiqueta correcta
        y_train.append(prediction.correction)

    return np.array(X_train), np.array(y_train)


def create_digit_image(digit, filename):
    # Crear una imagen en blanco
    img = Image.new('L', (28, 28), color=255)  # Fondo blanco
    d = ImageDraw.Draw(img)

    # Puedes cargar una fuente o usar una predeterminada
    font = ImageFont.load_default()

    # Dibujar el número en la imagen
    d.text((5, 5), str(digit), fill=0, font=font)

    # Redimensionar la imagen
    img = img.resize((28, 28), Image.LANCZOS)

    # Guardar la imagen
    img.save(filename)


def generate_number_image(request):
    form = NumberInputForm()
    success_message = None
    error_message = None

    if request.method == 'POST':
        form = NumberInputForm(request.POST)
        if form.is_valid():
            number = form.cleaned_data['number']
            image_filename = os.path.join(IMAGE_DIR, f'digit_{number}.png')
            create_digit_image(number, image_filename)
            success_message = f'Imagen del número {number} creada y guardada exitosamente.'
        else:
            error_message = "Por favor, ingresa un número válido entre 0 y 9."

    return render(request, 'classifier/generate_image.html', {
        'form': form,
        'success_message': success_message,
        'error_message': error_message
    })
