import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
for i in range(1,21):
    # C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Abnormal  - Aortic Stenosis (AS)\Abnormal  - Aortic Stenosis (AS)_combined_1.wav
    # C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Abnormal - Mitral Regurgitation (MR)\Abnormal - Mitral Regurgitation (MR)_combined_1.wav
    # C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Abnormal - Mitral Stenosis (MS)\Abnormal - Mitral Stenosis (MS)_combined_1.wav
    # C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Abnormal - Mitral Valve Prolapse (MVP)\Abnormal - Mitral Valve Prolapse (MVP)_combined_1.wav
    # C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Normal (N)\Normal (N)_combined_1.wav

    y, sr = librosa.load(f"Longer_data\\Normal (N)\\Normal (N)_combined_{i}.wav")

    # Compute Mel spectrogram

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to dB for better visualization

    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Display the spectrogram

    plt.figure(figsize=(20, 10))

    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram figure {i}')
    plt.show()


import librosa.display

def generate_mel_spectrogram(audio_path, sr=22050, n_fft=4096, hop_length=256, n_mels=256):
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr)
    
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Plot and save
    plt.figure(figsize=(20, 10))
    librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Enhanced Mel Spectrogram')
    plt.tight_layout()
    plt.show()
    return log_mel_spectrogram

# Example usage
# spectrogram = generate_mel_spectrogram("Longer_data\\Abnormal - Mitral Regurgitation (MR)\\Abnormal - Mitral Regurgitation (MR)_combined_1.wav")

