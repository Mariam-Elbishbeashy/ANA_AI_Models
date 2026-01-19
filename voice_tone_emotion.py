import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import sounddevice as sd
import joblib
import warnings

warnings.filterwarnings('ignore')

# ======================= CONFIG =======================
DATA_DIR = r"data\processed_audio"
EMOTIONS = ['angry', 'happy', 'sad', 'neutral', 'fear', 'surprise']
SAMPLE_RATE = 22050
DURATION = 3  
N_MFCC = 13
PCA_COMPONENTS = 50
KNN_NEIGHBORS = 7

# ======================= FEATURE EXTRACTION =======================
def extract_features_from_audio(audio, sr=SAMPLE_RATE):
    
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr).T, axis=0)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio).T, axis=0)

    return np.concatenate([
        mfccs, chroma, mel, zcr, rms,
        spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness
    ])


def extract_features(file_path):
   
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        return extract_features_from_audio(audio, sr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# ======================= LOAD DATA =======================
def load_dataset(data_dir=DATA_DIR, emotions=EMOTIONS):
    
    X, y = [], []
    print("Loading dataset and extracting features...")
    for emotion in emotions:
        folder = os.path.join(data_dir, emotion)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith((".wav", ".mp3")):
                file_path = os.path.join(folder, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion)
    return np.array(X), np.array(y)

# ======================= MODEL TRAINING =======================
def train_model(X, y):

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=PCA_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, weights='distance')
    knn.fit(X_train_pca, y_train)

    return knn, scaler, pca, le, X_test_pca, y_test

# ======================= MODEL EVALUATION =======================
def evaluate_model(knn, X_test, y_test, emotions=EMOTIONS):
   
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=emotions))

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(emotions)), emotions, rotation=45)
    plt.yticks(np.arange(len(emotions)), emotions)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# ======================= REAL-TIME PREDICTION =======================
def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    
    try:
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None

def predict_emotion_from_audio(audio, knn, scaler, pca, le):
    
    if audio is None:
        return None
    features = extract_features_from_audio(audio).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    prediction = knn.predict(features_pca)
    emotion = le.inverse_transform(prediction)[0]
    return emotion

def predict_emotion_from_file(file_path, knn, scaler, pca, le):
    
    features = extract_features(file_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    prediction = knn.predict(features_pca)
    emotion = le.inverse_transform(prediction)[0]
    print(f"Predicted Emotion from file: {emotion}")
    return emotion

# ======================= MAIN =======================
if __name__ == "__main__":
    X, y = load_dataset()
    knn, scaler, pca, le, X_test, y_test = train_model(X, y)

  
    joblib.dump(scaler, 'scaler2.pkl')
    joblib.dump(pca, 'pca2.pkl')
    joblib.dump(knn, 'knn2.pkl')
    joblib.dump(le, 'label2_encoder.pkl')
    print("Models saved successfully!")
    evaluate_model(knn, X_test, y_test)
