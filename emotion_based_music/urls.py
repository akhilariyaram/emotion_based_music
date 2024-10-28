from django.contrib import admin
from django.urls import include, path
from detector.views import detect_emotion  # Ensure you import the detect_emotion view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detector.urls')),  # Added a comma here
    path('detect-emotion/', detect_emotion, name='detect_emotion'),
]
