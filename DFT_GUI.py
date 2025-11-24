# FRONTEND GUI: DFT + Spectrogram Visualization
import tkinter as tk                           # Import modul tkinter untuk GUI
from tkinter import filedialog, messagebox     # Import dialog file dan pesan error
import numpy as np                             # Import NumPy untuk operasi numerik
import matplotlib.pyplot as plt                # Import Matplotlib untuk plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Integrasi Matplotlib dengan Tkinter
from Backend_DFT import analyze_audio, compute_fft  # Import fungsi analisis audio dari backend
from scipy.signal import chirp                 # Import fungsi chirp untuk membuat sinyal uji
import sounddevice as sd                       # Import sounddevice untuk playback audio
from scipy.io.wavfile import write             # Import fungsi write untuk menyimpan file WAV

# KELAS GUI
class DFTSpectrogramGUI:                       # Definisi kelas GUI utama
    def __init__(self, root):                  # Konstruktor kelas
        self.root = root                       # Simpan root Tkinter
        self.root.title("🎵 DFT + Spectrogram Analyzer")   # Judul jendela
        self.root.geometry("1300x850")         # Ukuran jendela
        self.root.configure(bg="#1E1E2F")      # Warna latar belakang utama

        # Frame kiri untuk tombol kontrol
        button_frame = tk.Frame(root, bg="#2C2C3E", bd=2, relief="ridge")  # Buat frame kiri
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)       # Atur posisi frame

        title_label = tk.Label(                # Label judul toolkit
            button_frame, text="🎧 AUDIO TOOLKIT", bg="#2C2C3E",
            fg="white", font=("Century Gothic", 14, "bold")
        )
        title_label.pack(pady=(10, 25))        # Atur posisi label

        # Gaya tombol standar
        def create_button(text, color, command):   # Fungsi untuk membuat tombol dengan gaya seragam
            btn = tk.Button(
                button_frame, text=text, bg=color, fg="white",
                activebackground="#57CC99", activeforeground="black",
                font=("Segoe UI", 10, "bold"), width=20, height=2,
                relief="flat", cursor="hand2", command=command
            )
            btn.pack(pady=8)                   # Atur jarak antar tombol
            return btn

        # Tombol-tombol
        create_button("🎼 Load WAV File", "#6A5ACD", self.load_wav)        # Tombol untuk load file WAV
        create_button("🎹 Generate Test Signal", "#4682B4", self.generate_signal) # Tombol buat sinyal uji
        create_button("🚪 Exit", "#C94C4C", self.root.quit)                # Tombol keluar aplikasi

        # FRAME KANAN: Grafik
        self.fig, (self.ax_mag, self.ax_phase, self.ax_spec) = plt.subplots(3, 1, figsize=(9, 8)) # Buat 3 subplot
        self.fig.subplots_adjust(hspace=0.4)   # Atur jarak antar subplot
        self.fig.patch.set_facecolor("#1E1E2F") # Warna latar belakang figure

        # Desain setiap subplot
        for ax, title in zip(
            [self.ax_mag, self.ax_phase, self.ax_spec],
            ["Amplitude Spectrum", "Phase Spectrum", "Spectrogram"]
        ):
            ax.set_facecolor("#0D0D0D")        # Warna latar belakang subplot
            ax.set_title(title, color="#00FFFF", fontsize=12, weight="bold") # Judul subplot
            ax.tick_params(colors="white")     # Warna sumbu putih
            ax.grid(alpha=0.3, color="gray", linestyle="--") # Tambah grid halus

        # Integrasi figure dengan tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=root) # Integrasi Matplotlib ke Tkinter
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True) # Tampilkan canvas

    # Fungsi: Load WAV
    def load_wav(self):                        # Fungsi untuk memuat file WAV
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")]) # Dialog pilih file
        if not file_path:                      # Jika tidak ada file dipilih
            return
        try:
            result = analyze_audio(file_path)  # Analisis file audio
            self.plot_all(result)              # Plot hasil analisis
        except Exception as e:                 # Jika error
            messagebox.showerror("Error", f"Gagal membaca file:\n{e}") # Tampilkan pesan error

    # Fungsi: Generate sinyal chirp (uji)
    def generate_signal(self):                 # Fungsi untuk membuat sinyal uji
        fs = 44100                             # Frekuensi sampling
        t = np.linspace(0, 5, fs * 5)          # Waktu 5 detik
        signal = chirp(t, f0=200, f1=10000, t1=5, method='linear') # Buat sinyal chirp
        write("test_signal.wav", fs, (signal * 32767).astype(np.int16)) # Simpan ke file WAV
        sd.play(signal, fs)                    # Putar sinyal
        result = analyze_audio("test_signal.wav") # Analisis sinyal uji
        self.plot_all(result)                  # Plot hasil analisis

    # Fungsi: Plot hasil (FFT + Spectrogram)
    def plot_all(self, result):                # Fungsi untuk menampilkan hasil analisis
        signal = result["signal"]              # Ambil sinyal
        sr = result["sample_rate"]             # Ambil sample rate
        fft_result = compute_fft(signal)       # Hitung FFT
        N = len(fft_result)                    # Panjang FFT
        freq = np.fft.fftfreq(N, 1 / sr)       # Hitung frekuensi

        # Bersihkan subplot sebelum menggambar ulang
        for ax in [self.ax_mag, self.ax_phase, self.ax_spec]:
            ax.clear()                         # Bersihkan isi subplot
            ax.set_facecolor("#0D0D0D")        # Warna latar belakang subplot
            ax.tick_params(colors="white")     # Warna sumbu putih

        # Plot Amplitude Spectrum
        self.ax_mag.plot(freq[:N // 2], np.abs(fft_result[:N // 2]), color="#00CED1") # Plot amplitudo
        self.ax_mag.set_title("DFT Amplitude Spectrum", color="#00FFFF", fontsize=11, weight="bold")
        self.ax_mag.set_xlabel("Frequency (Hz)", color="white")
        self.ax_mag.set_ylabel("Amplitude", color="white")

        # Plot Phase Spectrum
        phase = np.angle(fft_result)           # Hitung fase
        self.ax_phase.plot(freq[:N // 2], phase[:N // 2], color="#BA55D3") # Plot fase
        self.ax_phase.set_title("DFT Phase Spectrum", color="#00FFFF", fontsize=11, weight="bold")
        self.ax_phase.set_xlabel("Frequency (Hz)", color="white")
        self.ax_phase.set_ylabel("Phase (radians)", color="white")

        # Plot Spectrogram
        f = result["frequencies"]              # Ambil frekuensi dari spectrogram
        t = result["times"]                    # Ambil waktu dari spectrogram
        Sxx = result["spectrogram"]            # Ambil data spectrogram
        pcm = self.ax_spec.pcolormesh(
            t, f, 20 * np.log10(Sxx + 1e-6),   # Konversi ke dB
            shading="gouraud", cmap="plasma"   # Warna plasma
        )
        self.ax_spec.set_title("Spectrogram", color="#00FFFF", fontsize=11, weight="bold")
        self.ax_spec.set_xlabel("Time (s)", color="white")
        self.ax_spec.set_ylabel("Frequency (Hz)", color="white")

        # Colorbar
        cbar = self.fig.colorbar(pcm, ax=self.ax_spec) # Tambah colorbar
        cbar.set_label("Power/frequency (dB/Hz)", color="white") # Label colorbar
        cbar.ax.yaxis.set_tick_params(color="white")             # Warna sumbu colorbar
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white") # Warna label tick

        self.canvas.draw()                     # Render ulang canvas

# Main Program
if __name__ == "__main__":                     # Entry point program
    root = tk.Tk()                             # Buat root Tkinter
    app = DFTSpectrogramGUI(root)              # Inisialisasi aplikasi GUI
    root.mainloop()                            # Jalankan loop utama Tkinter
    