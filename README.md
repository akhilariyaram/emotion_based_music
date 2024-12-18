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
pip install absl-py==2.1.0 asgiref==3.8.1 astunparse==1.6.3 certifi==2024.8.30 channels==4.2.0 charset-normalizer==3.4.0 Django==5.1.4 flatbuffers==24.3.25 gast==0.6.0 google-pasta==0.2.0 grpcio==1.68.1 h5py==3.12.1 idna==3.10 keras==3.7.0 libclang==18.1.1 Markdown==3.7 markdown-it-py==3.0.0 MarkupSafe==3.0.2 mdurl==0.1.2 ml-dtypes==0.4.1 namex==0.0.8 numpy==2.0.2 opencv-python==4.10.0.84 opt_einsum==3.4.0 optree==0.13.1 packaging==24.2 pandas==2.2.3 pillow==11.0.0 pip==22.3 protobuf==5.29.1 Pygments==2.18.0 python-dateutil==2.9.0.post0 pytz==2024.2 requests==2.32.3 rich==13.9.4 setuptools==65.5.0 six==1.17.0 sqlparse==0.5.3 tensorboard==2.18.0 tensorboard-data-server==0.7.2 tensorflow==2.18.0 tensorflow_intel==2.18.0 tensorflow-io-gcs-filesystem==0.31.0 termcolor==2.5.0 typing_extensions==4.12.2 tzdata==2024.2 urllib3==2.2.3 Werkzeug==3.1.3 wheel==0.45.1 wrapt==1.17.0

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







