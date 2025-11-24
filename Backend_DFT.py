import numpy as np                  # Import library NumPy untuk operasi numerik
from scipy.io import wavfile        # Import modul wavfile untuk membaca file audio WAV
from scipy.signal import spectrogram # Import fungsi spectrogram untuk analisis frekuensi-waktu

# Fungsi: Baca file WAV
def read_audio(file_path):          # Definisi fungsi untuk membaca file audio
    sample_rate, data = wavfile.read(file_path)  # Membaca file WAV, menghasilkan sample rate dan data

    # Jika stereo (2 channel), ambil channel pertama
    if data.ndim > 1:               # Mengecek apakah data memiliki lebih dari 1 channel
        data = data[:, 0]           # Jika ya, ambil channel pertama saja

    # Normalisasi ke rentang -1.0 ... 1.0
    data = data.astype(np.float32)  # Konversi data ke tipe float32
    max_val = np.max(np.abs(data))  # Cari nilai maksimum absolut dari sinyal
    if max_val > 0:                 # Jika nilai maksimum lebih besar dari nol
        data /= max_val             # Normalisasi sinyal agar berada di rentang -1.0 sampai 1.0
    return sample_rate, data        # Mengembalikan sample rate dan data audio

# Fungsi: Hitung FFT (Fast Fourier Transform)
def compute_fft(signal):            # Definisi fungsi untuk menghitung FFTS
    return np.fft.fft(signal)       # Menghitung FFT dari sinyal input

# Fungsi: Hitung Spectrogram
def compute_spectrogram(signal, sample_rate):   # Definisi fungsi untuk menghitung spectrogram
    f, t, Sxx = spectrogram(signal, sample_rate) # Hitung frekuensi, waktu, dan spektrum daya
    return f, t, Sxx                # Mengembalikan hasil spectrogram

# Fungsi: Analisis Audio (gabungan semua)
def analyze_audio(file_path):       # Definisi fungsi utama untuk analisis audio
    sr, signal = read_audio(file_path)          # Membaca file audio dan mendapatkan sample rate serta sinyal
    fft_result = compute_fft(signal)            # Menghitung FFT dari sinyal
    f, t, Sxx = compute_spectrogram(signal, sr) # Menghitung spectrogram dari sinyal

    return {                        # Mengembalikan hasil analisis dalam bentuk dictionary
        "sample_rate": sr,          # Menyimpan sample rate
        "signal": signal,           # Menyimpan sinyal audio
        "fft_result": fft_result,   # Menyimpan hasil FFT
        "frequencies": f,           # Menyimpan array frekuensi dari spectrogram
        "times": t,                 # Menyimpan array waktu dari spectrogram
        "spectrogram": Sxx          # Menyimpan hasil spektrum daya (spectrogram)
    }