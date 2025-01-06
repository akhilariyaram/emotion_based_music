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
git clone <repository_url>
cd <repository_name>

Install the requirements
```bash
pip install numpy==1.23.5 pandas==1.5.3 django==4.2.9 tensorflow==2.14.0 opencv-python==4.7.0.72 keras==2.14.0
```
To run the Django project
```bash
python manage.py runserver
```
To stop the django application

click ctrl+c in cmd


Created ML model using ESSENTIALS\EMOTION DETECTION MODELS\fervi-notebook.ipynb this file.

Categorised songs into 4 categories using ESSENTIALS\EMOTION DETECTION MODELS\musiccategorization.ipynb this file




