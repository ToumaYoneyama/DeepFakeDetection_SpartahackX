import sys
import librosa
import numpy as np
import pickle
from moviepy import VideoFileClip

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = [np.mean(mfcc) for mfcc in mfccs]

    features = np.hstack([
        chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, *mfcc_means
    ])

    return features

# Function to extract audio from MP4 and save as WAV
def extract_audio_from_mp4(mp4_path, wav_path):
    # Load the video file
    video = VideoFileClip(mp4_path)

    # Extract audio
    audio = video.audio

    # Save audio as WAV
    audio.write_audiofile(wav_path)

    # Close the video file
    video.close()

def predict_deepfake(file_path):
    # Load the SVM model
    with open('/home/tyoneyam/spartahackX/resources/scaler.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('/home/tyoneyam/spartahackX/resources/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Check if the uploaded file is an MP4
    if file_path.endswith('.mp4'):
        # Extract audio from MP4 and save as WAV
        wav_path = file_path.replace('.mp4', '.wav')
        extract_audio_from_mp4(file_path, wav_path)

        # Use the extracted WAV file for feature extraction
        file_path = wav_path

    features = extract_features(file_path)
    scaled_features = scaler.transform([features])
    prediction = model.predict([scaled_features])
    
    return 'Fake' if prediction[0] == 1 else 'Real'

if __name__ == "__main__":
    file_path = sys.argv[1]
    result = predict_deepfake(file_path)
    print(result)