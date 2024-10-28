from django.urls import path
from .views import upload_image, live_emotion_detection

urlpatterns = [
    path('', upload_image, name='upload_image'),
    path('live/', live_emotion_detection, name='live_emotion_detection'),
]
