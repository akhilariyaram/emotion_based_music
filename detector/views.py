import os
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.shortcuts import render

def live_emotion_detection(request):
    return render(request, 'live_detection.html')
import json
from django.http import JsonResponse

def detect_emotion(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)
        
        # Preprocess and predict
        img = preprocess_image(file_path)
        pred = model.predict(img)
        pred_label = label[pred.argmax()]
        
        # Map the detected emotion to music label
        music_label = emotion_mapping.get(pred_label, 'Chill')  # Default to 'Chill' if not found

        # Get random songs for the mapped music label
        filtered_songs = music_df[music_df['label'] == music_label]
        print(f"Filtered songs for label '{music_label}':", filtered_songs)  # Debugging line

        if filtered_songs.empty:
            return JsonResponse({'error': 'No songs found for the detected emotion'}, status=404)

        song_links = filtered_songs.sample(n=5)['id'].tolist()
        print(f"Song links: {song_links}")  # Debugging line

        return JsonResponse({'emotion': pred_label, 'song_links': song_links})

    return JsonResponse({'error': 'Invalid request'}, status=400)



# Load your emotion detection model
model = load_model("emotiondetector.h5")

# Label mapping
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load music data
#music_data_path = r'D:\\emotion_based_music\\detector\\Updated_ClassifiedMusicData.csv' # Update this path
music_data_path = r"D:\\emotion_based_music\\detector\\ClassifiedMusicData.csv"
music_df = pd.read_csv(music_data_path)

# Emotion mapping
emotion_mapping = {
    'happy': 'Cheerful',
    'sad': 'Chill',
    'angry': 'Energetic',
    'fear': 'Chill',
    'surprise': 'Cheerful',
    'neutral': 'Chill',
    'disgust': 'Chill',
}

def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    feature = img_to_array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)
        
        # Preprocess and predict
        img = preprocess_image(file_path)
        pred = model.predict(img)
        pred_label = label[pred.argmax()]
        
        # Map the detected emotion to music label
        music_label = emotion_mapping.get(pred_label, 'Chill')  # Default to 'Chill' if not found

        # Get 5 random songs for the mapped music label
        filtered_songs = music_df[music_df['label'] == music_label].sample(n=25)

        # Create a list of song links (assuming you have a column with URLs)
        song_links = filtered_songs['id'].tolist()  # Ensure this corresponds to your DataFrame

        return render(request, 'result.html', {
            'emotion': pred_label,
            'image_path': fs.url(filename),
            'song_links': song_links  # Use 'song_links' for your template
        })

    return render(request, 'upload.html')

