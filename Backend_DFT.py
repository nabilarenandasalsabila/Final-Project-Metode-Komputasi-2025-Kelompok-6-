import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

# Fungsi: Baca file WAV
def read_audio(file_path):
    sample_rate, data = wavfile.read(file_path)

    # Jika stereo (2 channel), ambil channel pertama
    if data.ndim > 1:
        data = data[:, 0]

    # Normalisasi ke rentang -1.0 ... 1.0
    data = data.astype(np.float32)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data /= max_val
    return sample_rate, data

# Fungsi: Hitung FFT (Fast Fourier Transform)
def compute_fft(signal):
    return np.fft.fft(signal)

# Fungsi: Hitung Spectrogram
def compute_spectrogram(signal, sample_rate):
    f, t, Sxx = spectrogram(signal, sample_rate)
    return f, t, Sxx

# Fungsi: Analisis Audio (gabungan semua)
def analyze_audio(file_path):
    sr, signal = read_audio(file_path)
    fft_result = compute_fft(signal)
    f, t, Sxx = compute_spectrogram(signal, sr)

    return {
        "sample_rate": sr,
        "signal": signal,
        "fft_result": fft_result,
        "frequencies": f,
        "times": t,
        "spectrogram": Sxx
    }