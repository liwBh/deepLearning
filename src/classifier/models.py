from django.db import models

class Prediction(models.Model):
    image = models.ImageField(upload_to='images/')  # Almacena la imagen cargada
    prediction = models.IntegerField()  # predicción del modelo
    correction = models.IntegerField(null=True, blank=True)  # corrección si es necesaria
    fail = models.BooleanField(default=True)  # Indica si la predicción fue corregida o no
    created_at = models.DateTimeField(auto_now_add=True)  # Fecha de la predicción

    def __str__(self):
        return f"Predicción: {self.prediction}, Corregido: {self.correction if self.correction else 'No'}"
