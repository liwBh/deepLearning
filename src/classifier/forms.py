from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        widget=forms.ClearableFileInput(attrs={'class': 'form-control form-control-sm'})
    )


class CorrectionForm(forms.Form):
    correction = forms.IntegerField(
        min_value=0,
        max_value=9,
        widget=forms.NumberInput(attrs={'class': 'form-control form-control-sm'}),
        label='Ingresa el numero correcto (0-9)'
    )