# Emotion-Based Music Recommendation System 🎵😊

This project integrates **emotion detection** with **music recommendation** to provide personalized song suggestions based on your mood. It uses facial emotion recognition and categorized Spotify data to recommend songs tailored to the detected emotions.

---

## Datasets Used:
- **FER2013** and **CK+ Dataset**: For image-based emotion detection.
- **Spotify Data**: For categorizing music into emotional categories.

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/akhilariyaram/emotion_based_music.git
cd emotion_based_music
```
### Install the requirements
```bash
pip install numpy==1.23.5 pandas==1.5.3 django==4.2.9 tensorflow==2.14.0 opencv-python==4.7.0.72 keras==2.14.0
```
### To run the Django project
```bash
python manage.py runserver
```
### To stop the django application

click ctrl+c in cmd

### Machine Learning Model

Created ML model using ESSENTIALS\EMOTION DETECTION MODELS\fervi-notebook.ipynb this file.

This model leverages a Convolutional Neural Network (CNN) architecture, specifically trained on the FER2013 and CK+ datasets. The model is capable of detecting various facial emotions such as happiness, sadness, surprise, anger, etc. After training and fine-tuning, the model achieved an impressive accuracy of 82%. This high accuracy ensures that the system can reliably predict emotions from facial expressions, forming the core of the emotion-based music recommendation system.

## And the model has reached an accuracy of 82 percent

### Music Categorization 

Categorised songs into 4 categories using ESSENTIALS\EMOTION DETECTION MODELS\musiccategorization.ipynb this file.

In this file, music tracks from the Spotify dataset were analyzed based on various features like danceability, acousticness, and loudness. Using a machine learning model, the songs were classified into one of the following four categories:

Energetic
Chill
Romantic
Cheerful
These categories align with the four primary emotional states detected by the facial emotion recognition model. Once the songs were categorized, they could then be recommended based on the detected emotion from the user's face.






