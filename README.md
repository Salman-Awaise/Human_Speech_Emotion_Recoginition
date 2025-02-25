# Human_Speech_Emotion_Recoginition

1. Project Overview
This project aims to recognize human emotions from speech recordings using audio signal processing and machine learning techniques. The system processes audio files, extracts key features, and classifies them into different emotion categories.

2. Dataset
The project uses speech samples containing emotional variations.
The dataset consists of .wav files labeled with different emotions.
3. Audio Processing Techniques
The notebook applies the following preprocessing steps on raw audio:

Loading & Visualizing Audio

Uses librosa and pydub to load and analyze speech waveforms.
Displays the waveform of the original audio.
Normalization

Applies volume normalization to make all audio samples comparable in intensity.
Trimming Silence

Removes leading and trailing silence to improve feature extraction.
Padding

Ensures that all audio clips have the same duration.
Noise Reduction

Uses the noisereduce library to remove background noise for clearer speech signals.
4. Feature Extraction
The following audio features are extracted to differentiate between emotions:

Root Mean Square Energy (RMS) – Measures loudness variations.
Zero Crossing Rate (ZCR) – Detects frequency fluctuations.
Mel-Frequency Cepstral Coefficients (MFCCs) – Captures speech characteristics.
Chroma Features – Identifies musical tones in speech.
Spectral Contrast – Highlights differences between frequency components.
5. Model Training
The extracted features are used to train a machine learning model for emotion classification.
Possible models:
Random Forest Classifier
Support Vector Machine (SVM)
Neural Networks (LSTM, CNNs for deep learning)
6. Installation & Dependencies
Prerequisites
Python 3.x
Jupyter Notebook or Google Colab
Required Libraries
Install dependencies using:

sh
Copy
Edit
pip install pydub noisereduce librosa numpy pandas matplotlib seaborn scikit-learn
7. Running the Notebook
Open Human_Speech_Emotion_Recoginition.ipynb in Jupyter Notebook or Google Colab.
Ensure that the dataset (audio .wav files) is accessible.
Run all the cells to process audio, extract features, and classify emotions.
8. Results & Visualizations
Displays waveform plots before and after noise reduction.
Shows feature distributions for different emotions.
Provides classification accuracy and confusion matrices.
9. Future Improvements
Use deep learning models (CNNs, LSTMs) for better accuracy.
Train on larger datasets for robust emotion recognition.
Implement real-time emotion detection in applications.
