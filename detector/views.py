import os
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.http import JsonResponse
import cv2
from django.http import HttpResponse
from django.conf import settings

from time import sleep
t=r"D:\\Projects\\emotion_based_music\\detector\\face_emotion.h5"
emotion_model = load_model(settings.MODEL_PATH)
# Load the music dataset
temp=r"D:\\Projects\\emotion_based_music\\detector\\musicData.csv"
mood_music = pd.read_csv(settings.CSV_PATH)
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Function to filter songs based on mood
music_data_path = r"D:\\emotion_based_music\\detector\\ClassifiedMusicData.csv"
music_df = pd.read_csv(settings.CSV)
emotion_mapping = {
    'happy': 'Cheerful',
    'sad': 'Chill',
    'angry': 'Energetic',
    'fear': 'Chill',
    'surprise': 'Romantic',
    'neutral': 'Chill',
    'disgust': 'Chill',
}
def music_results(n):
    if n in (0, 1, 2):
        # For angry, disgust, fear
        filter1 = mood_music['mood'] == 'Chill'
    elif n in (3, 4):
        # For happy, neutral
        filter1 = mood_music['mood'] == 'energetic'
    elif n == 5:
        # For sad
        filter1 = mood_music['mood'] == 'cheerful'
    elif n == 6:
        # For surprise
        filter1 = mood_music['mood'] == 'romantic'
    else:
        return pd.DataFrame()  

    f1 = mood_music.where(filter1).dropna()
    f2 = f1.sample(n=5)
    f2.reset_index(inplace=True)
    return f2 

def live_emotion_detection(request):
    emotion_dict = {
        0: "Angry", 
        1: "Disgusted", 
        2: "Fearful", 
        3: "Happy", 
        4: "Neutral", 
        5: "Sad", 
        6: "Surprised"
    }
    

    webcam = cv2.VideoCapture(0)
    sleep(2)  

    data = None 
    result = None  

    while True:
        try:
            check, image = webcam.read()
            if not check: 
                print("Failed to grab frame")
                break

            cv2.imshow("Capturing", image)
            key = cv2.waitKey(1)

            if key == ord('s'):  
                
                cv2.imwrite('static/captured_image.jpg', image)  
                print("Image captured and saved as 'captured_image.jpg'")

               
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                temp1=r"D:\\Projects\\emotion_based_music\\detector\\haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(settings.FACADEPATH)
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = image[y:y + h, x:x + w]
                    gray_face = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                    img_resized = cv2.resize(gray_face, (48, 48))
                    img = np.array(img_resized).reshape(1, 48, 48, 1) / 255.0 
                    predict_x = emotion_model.predict(img)
                    pred_label = label[predict_x.argmax()]
                    music_label = emotion_mapping.get(pred_label, 'Chill')
                    filtered_songs = music_df[music_df['label'] == music_label].sample(n=25)

                    song_links = filtered_songs['id'].tolist()
                    result = np.argmax(predict_x, axis=1)
                    print(emotion_dict[result[0]])

                   
                    data = music_results(result[0])
                    print(data[['name', 'artist',"id"]])  

            elif key == ord('q'):  
                break

        except KeyboardInterrupt:
            print("Turning off camera.")
            break

    webcam.release()
    cv2.destroyAllWindows()

    context = {
        "songs": song_links,
        "mood": emotion_dict[result[0]] if result is not None else "Unknown"
    }

    return render(request, 'live_detection.html', context)



model = load_model("emotiondetector.h5")

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']



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
        
        img = preprocess_image(file_path)
        pred = model.predict(img)
        pred_label = label[pred.argmax()]
        
        music_label = emotion_mapping.get(pred_label, 'Chill')

        filtered_songs = music_df[music_df['label'] == music_label].sample(n=25)

        song_links = filtered_songs['id'].tolist()

        return render(request, 'result.html', {
            'emotion': pred_label,
            'image_path': fs.url(filename),
            'song_links': song_links
        })

    return render(request, 'upload.html')
