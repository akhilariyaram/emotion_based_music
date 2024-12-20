# Emotion-Based Music Recommendation System
This is a Django-based web application that recommends music based on the emotion detected from uploaded images or live webcam feed. The system uses a pre-trained emotion detection model to classify emotions and map them to music genres. The application allows users to upload images for emotion detection or use live webcam detection.

# Features
Image Upload for Emotion Prediction: Users can upload an image, and the model will predict the emotion. Based on the predicted emotion, songs from a relevant genre are recommended.
Live Emotion Detection: Users can start a live webcam session to detect emotions in real-time. When a face is detected, the system predicts the emotion and recommends songs accordingly.
Emotion-Music Mapping: Each emotion is mapped to a specific mood (e.g., happy → cheerful), and songs are recommended based on the mood.
Spotify Integration: Recommended songs are displayed as embedded Spotify players that can be played directly from the web page.
Prerequisites
Python 3.x
Django 5.1 or higher
TensorFlow
OpenCV
Pandas
Numpy
Music Data CSV File
Emotion Detection Model (h5 file)

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt


```

## To Run
# Open Project Folder 

![Screenshot 2024-12-16 132642](https://github.com/user-attachments/assets/57d6310d-3a40-4786-ab74-f1af645dc57e)

# Enter cmd Here and click enter
![Screenshot 2024-12-16 132850](https://github.com/user-attachments/assets/c6be03ad-fa9c-468d-adb5-8bfa273d9b4a)

# Then a command prompt will be opened

Then enter below command there and click enter it takes a couple of moments and after successful running it shows as

```bash
python manage.py runserver
```

![Screenshot 2024-12-16 133231](https://github.com/user-attachments/assets/87fbe2ee-c8f3-4ee6-a477-5abc7b8d62be)


Then open the link  http://127.0.0.1:8000/  by ctrl+click 

It will be opened in a browser

Like This 

![Screenshot 2024-12-16 133351](https://github.com/user-attachments/assets/b4c83b94-8ee3-4c8e-ba28-df3d0289eba0)


If u upload an image there the result page looks like this 

![Screenshot 2024-12-16 134223](https://github.com/user-attachments/assets/8891dd01-36a5-4ad8-b8cb-0c2e61747969)


If u click this live detection it takes couple of moments a camera will be opened if u click s an image will be captured and emotion will be detected as below

![Screenshot 2024-12-16 134803](https://github.com/user-attachments/assets/c25430fa-15f9-4b48-a223-28f83cc02566)

if u click s an new image will be captured and emotion will be detected and shown in cmd 

Click    "q"   to exit camera

final result will be as for live detection as 

![Screenshot 2024-12-16 134223](https://github.com/user-attachments/assets/b9cd6c66-8c3c-4162-b190-a3fe5826bc62)







