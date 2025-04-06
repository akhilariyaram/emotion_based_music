import os
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from django.http import JsonResponse
import cv2
from django.conf import settings

model=load_model(settings.MODEL_PATH)
label = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
music_data_path = r"D:\\emotion_based_music\\detector\\ClassifiedMusicData.csv"
music_df = pd.read_csv(settings.CSV)

# Emotion to music mapping
emotion_mapping = {
    'happy': 'Energetic',
    'sad': 'Cheerful',
    'angry': 'Calm',
    'fear': 'Calm',
    'surprise': 'Chillful',
    'neutral': 'Chillful',
    'disgust': 'Energetic',
}

# Preprocessing function to detect faces
def preprocess_image(image_path):
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image not loaded. Check the file path.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.shape == (48, 48):
        print("Input image is already 48x48.")
        feature = img_to_array(gray).reshape(1, 48, 48, 1) / 255.0
        return feature

    elif len(image.shape) == 2:
        feature = img_to_array(gray).reshape(1, 48, 48, 1) / 255.0
        return feature

    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
        return None

    preprocessed_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(roi_gray, (48, 48))
        normalized_face = resized_face / 255.0
        input_image = np.expand_dims(np.expand_dims(normalized_face, -1), 0)
        preprocessed_faces.append(input_image)

    return preprocessed_faces[0]
def set_emotion(request, pred_label):
    request.session['detected_emotion'] = pred_label

def get_emotion(request):
    return request.session.get('detected_emotion', 'neutral')

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()

        # Save the image with a fixed name to always overwrite the file
        filename = "uploaded_image.jpg"  # Fixed name for the uploaded image
        t=r"D:\Projects\aztask\emotion_based_music\emotion_based_music\static"+'/'+filename
        file_path = fs.location + '/' + filename
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(t):
            os.remove(t)
        # Save the new uploaded image with the same name
        fs.save(filename, uploaded_file)  # This saves the file to a directory with the same name
        fs.save(t,uploaded_file)
        img = preprocess_image(file_path)

        if img is None:
            # If no faces are detected, return Chill music and an appropriate message
            pred_label = 'neutral'  # Default emotion if no face is detected
            music_label = 'Energetic'   # Default label for Chill music
            filtered_songs = music_df[music_df['label'] == music_label]

            # Shuffle the filtered songs and select the top 35 random songs
            random_songs = filtered_songs.sample(n=10)  # This will give a different random sample every time


            song_links = random_songs['id'].tolist()  # Get the song links from the random selection

            return render(request, 'result.html', {
                'emotion': 'No faces detected (poor image quality)',  # Display message
                'image_path': fs.url(filename),
                'song_links': song_links,
                'label': music_label
            })
        
        # If faces are detected, proceed with emotion prediction
        pred = model.predict(img)
        
        pred_label = label[pred.argmax()]
        set_emotion(request, pred_label)
        music_label = emotion_mapping.get(pred_label, 'Energetic')
        filtered_songs = music_df[music_df['label'] == music_label]
        random_songs = filtered_songs.sample(n=10)  # This will give a different random sample every time

        song_links = random_songs['id'].head(10).tolist()  # Limit to the first 35 songs
        
        return render(request, 'result.html', {
            'emotion': pred_label,
            'image_path': t,
            'song_links': song_links,
            'label': music_label
        })

    return render(request, 'upload.html')

# View function for live emotion detection (if needed in the future)
def live_emotion_detection(request):
    return render(request, 'live_detection.html')

# Endpoint for detecting emotion from the uploaded image and returning song links
def detect_emotion(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()

        # Fixed filename to always save the file with the same name
        filename = "uploaded_image.jpg"  # Fixed name for the uploaded image
        file_path = fs.location + '/' + filename

        # Check if the file already exists and remove it
        if os.path.exists(file_path):
            os.remove(file_path)

        # Save the new uploaded image with the same name
        fs.save(filename, uploaded_file)  # This saves the file to a directory with the same name
        
        img = preprocess_image(file_path)

        if img is None:
            # If no faces are detected, return default Chill music and no emotion
            pred_label = 'neutral'
            music_label = 'Energetic'

            filtered_songs = music_df[music_df['label'] == music_label]
            if filtered_songs.empty:
                return JsonResponse({'error': 'No songs found for the detected emotion'}, status=404)

            random_songs = filtered_songs.sample(n=10)  # This will give a different random sample every time

            song_links = random_songs['id'].head(10).tolist()  # Limit to the first 35 songs

            return JsonResponse({'emotion': 'No faces detected', 'song_links': song_links})
        
        # Predict emotion if faces are detected
        pred = model.predict(img)
        pred_label = label[pred.argmax()]
        set_emotion(request, pred_label)
        music_label = emotion_mapping.get(pred_label, 'Energetic')

        filtered_songs = music_df[music_df['label'] == music_label]
        if filtered_songs.empty:
            return JsonResponse({'error': 'No songs found for the detected emotion'}, status=404)

        random_songs = filtered_songs.sample(n=10)  # This will give a different random sample every time

        song_links = random_songs['id'].head(10).tolist()  # Limit to the first 35 songs

        return JsonResponse({'emotion': pred_label, 'song_links': song_links})

    return JsonResponse({'error': 'Invalid request'}, status=400)

# Filter songs by language
from django.http import JsonResponse

from django.http import JsonResponse
import pandas as pd

def filter_songs(request):
    language = request.GET.get('language', 'all')

    # Get the detected emotion using the helper function
    detected_emotion = get_emotion(request)
    music_label = emotion_mapping.get(detected_emotion, 'Energetic')  # Default to 'Energetic' if not found

    # Filter songs based on language
    if language == 'all':
        filtered_songs = music_df[music_df['language'].str.lower().isin(['telugu', 'english', 'tamil'])]
    else:
        filtered_songs = music_df[music_df['language'].str.lower() == language.lower()]

    # Filter songs based on detected emotion (mood)
    filtered_songs = filtered_songs[filtered_songs['label'].str.lower() == music_label.lower()]

    # If no songs match, return an empty list
    if filtered_songs.empty:
        return JsonResponse({'song_links': []})

    # Sort songs alphabetically
    filtered_songs = filtered_songs.sort_values(by=['name'], ascending=True)

    # Select the first 10 songs (or fewer if there aren't enough)
    song_links = filtered_songs['id'].head(10).tolist()

    return JsonResponse({'song_links': song_links})
