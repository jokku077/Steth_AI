import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file

# C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Abnormal  - Aortic Stenosis (AS)\Abnormal  - Aortic Stenosis (AS)_combined_1.wav
# C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Abnormal - Mitral Regurgitation (MR)\Abnormal - Mitral Regurgitation (MR)_combined_1.wav
# C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Abnormal - Mitral Stenosis (MS)\Abnormal - Mitral Stenosis (MS)_combined_1.wav
# C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Abnormal - Mitral Valve Prolapse (MVP)\Abnormal - Mitral Valve Prolapse (MVP)_combined_1.wav
# C:\Users\jokku\Major Project\Model_w_longdata\Longer_data\Normal (N)\Normal (N)_combined_1.wav

y, sr = librosa.load(f"Longer_data\\Abnormal - Mitral Valve Prolapse (MVP)\\Abnormal - Mitral Valve Prolapse (MVP)_combined_12.wav")

# Compute Mel spectrogram

mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to dB for better visualization

mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Display the spectrogram

plt.figure(figsize=(20, 10))

librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel Spectrogram figure MR')
plt.show()