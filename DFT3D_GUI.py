import tkinter as tk                           # Import modul tkinter untuk GUI
from tkinter import filedialog, messagebox     # Import dialog file dan pesan pop-up
import numpy as np                             # Import NumPy untuk operasi numerik
import matplotlib.pyplot as plt                # Import Matplotlib untuk plotting grafik
from mpl_toolkits.mplot3d import Axes3D        # Import toolkit untuk membuat grafik 3D
from Backend_DFT import analyze_audio, compute_fft  # Import fungsi analisis audio dari backend
from scipy.signal import chirp                 # Import fungsi chirp untuk membuat sinyal uji
import sounddevice as sd                       # Import sounddevice untuk playback audio
from scipy.io.wavfile import write             # Import fungsi write untuk menyimpan file WAV

class DFTSpectrogramGUI:                       # Definisi kelas GUI utama
    def __init__(self, root):                  # Konstruktor kelas
        self.root = root                       # Simpan root Tkinter
        self.root.title("🎵 DFT + Spectrogram Analyzer (Interaktif 3D)") # Judul jendela GUI
        self.root.geometry("400x500")          # Ukuran jendela GUI
        self.root.configure(bg="#1E1E2F")      # Warna latar belakang utama

        button_frame = tk.Frame(root, bg="#2C2C3E", bd=2, relief="ridge") # Frame untuk tombol
        button_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)    # Atur posisi frame

        title_label = tk.Label(                # Label judul toolkit
            button_frame, text="🎧 AUDIO TOOLKIT", bg="#2C2C3E",
            fg="white", font=("Century Gothic", 14, "bold")
        )
        title_label.pack(pady=(10, 25))        # Atur posisi label

        def create_button(text, color, command):   # Fungsi untuk membuat tombol dengan gaya seragam
            btn = tk.Button(
                button_frame, text=text, bg=color, fg="white",
                activebackground="#57CC99", activeforeground="black",
                font=("Segoe UI", 10, "bold"), width=25, height=2,
                relief="flat", cursor="hand2", command=command
            )
            btn.pack(pady=8)                   # Atur jarak antar tombol
            return btn

        # Tombol-tombol untuk fungsi GUI
        create_button("🎼 Load WAV File", "#6A5ACD", self.load_wav)            # Tombol load file WAV
        create_button("🎹 Generate Test Signal", "#4682B4", self.generate_signal) # Tombol buat sinyal uji
        create_button("📈 Show Amplitude Spectrum", "#00CED1", self.show_amplitude_plot) # Tombol amplitudo
        create_button("📐 Show Phase Spectrum", "#BA55D3", self.show_phase_plot) # Tombol fase
        create_button("🌈 Show Spectrogram", "#FF6347", self.show_spectrogram_plot) # Tombol spectrogram
        create_button("🚪 Exit", "#C94C4C", self.root.quit)                    # Tombol keluar aplikasi

        self.result = None                      # Variabel untuk menyimpan hasil analisis audio

    def load_wav(self):                         # Fungsi untuk memuat file WAV
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")]) # Dialog pilih file
        if not file_path:                       # Jika tidak ada file dipilih
            return
        try:
            self.result = analyze_audio(file_path) # Analisis file audio
            messagebox.showinfo("Sukses", "File berhasil dimuat. Silakan pilih jenis grafik yang ingin ditampilkan.")
        except Exception as e:                  # Jika error
            messagebox.showerror("Error", f"Gagal membaca file:\n{e}") # Tampilkan pesan error

    def generate_signal(self):                  # Fungsi untuk membuat sinyal uji chirp
        fs = 44100                              # Frekuensi sampling
        t = np.linspace(0, 5, fs * 5)           # Waktu 5 detik
        signal = chirp(t, f0=200, f1=10000, t1=5, method='linear') # Buat sinyal chirp
        write("test_signal.wav", fs, (signal * 32767).astype(np.int16)) # Simpan ke file WAV
        sd.play(signal, fs)                     # Putar sinyal
        self.result = analyze_audio("test_signal.wav") # Analisis sinyal uji
        messagebox.showinfo("Sinyal Uji", "Sinyal chirp berhasil dibuat dan dimainkan.\nSilakan pilih jenis grafik yang ingin ditampilkan.")

    def show_amplitude_plot(self):              # Fungsi untuk menampilkan grafik amplitudo
        if not self.result:                     # Jika belum ada data
            messagebox.showwarning("Belum Ada Data", "Silakan load file WAV atau generate sinyal terlebih dahulu.")
            return

        signal = self.result["signal"]          # Ambil sinyal
        sr = self.result["sample_rate"]         # Ambil sample rate
        fft_result = compute_fft(signal)        # Hitung FFT
        N = len(fft_result)                     # Panjang FFT
        freq = np.fft.fftfreq(N, 1 / sr)        # Hitung frekuensi

        fig = plt.figure(figsize=(8, 6), facecolor='white') # Buat figure baru
        ax = fig.add_subplot(111, projection='3d', facecolor='white') # Subplot 3D
        x = freq[:N // 2]                       # Frekuensi positif
        y = np.zeros_like(x)                    # Sumbu referensi (nol)
        z = np.abs(fft_result[:N // 2])         # Amplitudo FFT
        ax.plot(x, y, z, color="#00CED1")       # Plot amplitudo dalam 3D
        ax.set_title("DFT Amplitude Spectrum", color="black", fontsize=12, weight="bold")
        ax.set_xlabel("Frequency (Hz)", color="black")
        ax.set_ylabel("Reference Axis", color="black")
        ax.set_zlabel("Amplitude", color="black")
        ax.auto_scale_xyz(x, y, z)              # Skala otomatis
        plt.tight_layout()                      # Atur layout
        plt.show()                              # Tampilkan grafik

    def show_phase_plot(self):                  # Fungsi untuk menampilkan grafik fase
        if not self.result:                     # Jika belum ada data
            messagebox.showwarning("Belum Ada Data", "Silakan load file WAV atau generate sinyal terlebih dahulu.")
            return

        signal = self.result["signal"]          # Ambil sinyal
        sr = self.result["sample_rate"]         # Ambil sample rate
        fft_result = compute_fft(signal)        # Hitung FFT
        N = len(fft_result)                     # Panjang FFT
        freq = np.fft.fftfreq(N, 1 / sr)        # Hitung frekuensi

        fig = plt.figure(figsize=(8, 6), facecolor='white') # Buat figure baru
        ax = fig.add_subplot(111, projection='3d', facecolor='white') # Subplot 3D
        x = freq[:N // 2]                       # Frekuensi positif
        y = np.zeros_like(x)                    # Sumbu referensi (nol)
        z_phase = np.angle(fft_result[:N // 2]) # Fase FFT
        ax.plot(x, y, z_phase, color="#BA55D3") # Plot fase dalam 3D
        ax.set_title("DFT Phase Spectrum", color="black", fontsize=12, weight="bold")
        ax.set_xlabel("Frequency (Hz)", color="black")
        ax.set_ylabel("Reference Axis", color="black")
        ax.set_zlabel("Phase (radians)", color="black")
        ax.auto_scale_xyz(x, y, z_phase)        # Skala otomatis
        plt.tight_layout()                      # Atur layout
        plt.show()                              # Tampilkan grafik

    def show_spectrogram_plot(self):            # Fungsi untuk menampilkan spectrogram 3D
        if not self.result:                     # Jika belum ada data
            messagebox.showwarning("Belum Ada Data", "Silakan load file WAV atau generate sinyal terlebih dahulu.")
            return

        f = self.result["frequencies"]          # Ambil array frekuensi
        t = self.result["times"]                # Ambil array waktu
        Sxx = self.result["spectrogram"]        # Ambil data spectrogram
        T, F = np.meshgrid(t, f)                # Buat grid waktu-frekuensi
        Z = 20 * np.log10(Sxx + 1e-6)           # Konversi ke dB

        fig = plt.figure(figsize=(8, 6), facecolor='white') # Buat figure baru
        ax = fig.add_subplot(111, projection='3d', facecolor='white') # Subplot 3D
        surf = ax.plot_surface(T, F, Z, cmap="plasma", edgecolor='none') # Plot permukaan spect
        ax.set_title("Spectrogram", color="black", fontsize=12, weight="bold") # Judul grafik spectrogram
        ax.set_xlabel("Time (s)", color="black")   # Label sumbu X (waktu)
        ax.set_ylabel("Frequency (Hz)", color="black") # Label sumbu Y (frekuensi)
        ax.set_zlabel("Power (dB)", color="black") # Label sumbu Z (daya dalam dB)
        ax.auto_scale_xyz(t, f, Z)                 # Skala otomatis untuk sumbu
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10) # Tambahkan colorbar untuk intensitas
        plt.tight_layout()                         # Atur layout agar rapi
        plt.show()                                 # Tampilkan grafik spectrogram 3D

# Main Program
if __name__ == "__main__":                        # Entry point program
    root = tk.Tk()                                # Buat root Tkinter
    app = DFTSpectrogramGUI(root)                 # Inisialisasi aplikasi GUI
    root.mainloop()                               # Jalankan loop utama Tkinter
    