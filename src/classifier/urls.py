from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('generate_image/', views.generate_number_image, name='generate_image'),

]
