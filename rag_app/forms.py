from django import forms
from .models import PDFDocument

class PDFUploadForm(forms.ModelForm):
    class Meta:
        model = PDFDocument
        fields = ['pdf_file']
        widgets = {
            'pdf_file': forms.ClearableFileInput(),
        }
