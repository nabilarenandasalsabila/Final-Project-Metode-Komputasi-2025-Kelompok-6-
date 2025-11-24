# FRONTEND GUI: DFT + Spectrogram Visualization 
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Backend_DFT import analyze_audio, compute_fft
from scipy.signal import chirp
import sounddevice as sd
from scipy.io.wavfile import write

# KELAS GUI 
class DFTSpectrogramGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 DFT + Spectrogram Analyzer")
        self.root.geometry("1300x850")
        self.root.configure(bg="#1E1E2F")  # Warna latar belakang utama (gelap elegan)

        # Frame kiri untuk tombol kontrol
        button_frame = tk.Frame(root, bg="#2C2C3E", bd=2, relief="ridge")
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)

        title_label = tk.Label(
            button_frame, text="🎧 AUDIO TOOLKIT", bg="#2C2C3E",
            fg="white", font=("Century Gothic", 14, "bold")
        )
        title_label.pack(pady=(10, 25))

        # Gaya tombol standar
        def create_button(text, color, command):
            btn = tk.Button(
                button_frame, text=text, bg=color, fg="white",
                activebackground="#57CC99", activeforeground="black",
                font=("Segoe UI", 10, "bold"), width=20, height=2,
                relief="flat", cursor="hand2", command=command
            )
            btn.pack(pady=8)
            return btn

        # Tombol-tombol
        create_button("🎼 Load WAV File", "#6A5ACD", self.load_wav)
        create_button("🎹 Generate Test Signal", "#4682B4", self.generate_signal)
        create_button("🚪 Exit", "#C94C4C", self.root.quit)

        # FRAME KANAN: Grafik
        self.fig, (self.ax_mag, self.ax_phase, self.ax_spec) = plt.subplots(3, 1, figsize=(9, 8))
        self.fig.subplots_adjust(hspace=0.4)
        self.fig.patch.set_facecolor("#1E1E2F")  # Latar figure disamakan dengan jendela

        # Desain setiap subplot
        for ax, title in zip(
            [self.ax_mag, self.ax_phase, self.ax_spec],
            ["Amplitude Spectrum", "Phase Spectrum", "Spectrogram"]
        ):
            
            ax.set_facecolor("#0D0D0D")  # Warna latar grafik
            ax.set_title(title, color="#00FFFF", fontsize=12, weight="bold")  # Warna teks terang
            ax.tick_params(colors="white")  # Warna sumbu putih
            ax.grid(alpha=0.3, color="gray", linestyle="--")  # Tambah grid halus

        # Integrasi figure dengan tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Fungsi: Load WAV
    def load_wav(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if not file_path:
            return
        try:
            result = analyze_audio(file_path)
            self.plot_all(result)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca file:\n{e}")

    # Fungsi: Generate sinyal chirp (uji)
    def generate_signal(self):
        fs = 44100
        t = np.linspace(0, 5, fs * 5)
        signal = chirp(t, f0=200, f1=10000, t1=5, method='linear')
        write("test_signal.wav", fs, (signal * 32767).astype(np.int16))
        sd.play(signal, fs)
        result = analyze_audio("test_signal.wav")
        self.plot_all(result)

    # Fungsi: Plot hasil (FFT + Spectrogram)
    def plot_all(self, result):
        signal = result["signal"]
        sr = result["sample_rate"]
        fft_result = compute_fft(signal)
        N = len(fft_result)
        freq = np.fft.fftfreq(N, 1 / sr)

        # Bersihkan subplot sebelum menggambar ulang
        for ax in [self.ax_mag, self.ax_phase, self.ax_spec]:
            ax.clear()
            ax.set_facecolor("#0D0D0D")
            ax.tick_params(colors="white")

        # Plot Amplitude Spectrum (ubah label dan judul)
        self.ax_mag.plot(freq[:N // 2], np.abs(fft_result[:N // 2]), color="#00CED1")
        self.ax_mag.set_title("DFT Amplitude Spectrum", color="#00FFFF", fontsize=11, weight="bold")
        self.ax_mag.set_xlabel("Frequency (Hz)", color="white")
        self.ax_mag.set_ylabel("Amplitude", color="white")

        # Plot Phase Spectrum
        phase = np.angle(fft_result)
        self.ax_phase.plot(freq[:N // 2], phase[:N // 2], color="#BA55D3")
        self.ax_phase.set_title("DFT Phase Spectrum", color="#00FFFF", fontsize=11, weight="bold")
        self.ax_phase.set_xlabel("Frequency (Hz)", color="white")
        self.ax_phase.set_ylabel("Phase (radians)", color="white")

        # Plot Spectrogram
        f = result["frequencies"]
        t = result["times"]
        Sxx = result["spectrogram"]
        pcm = self.ax_spec.pcolormesh(
            t, f, 20 * np.log10(Sxx + 1e-6),
            shading="gouraud", cmap="plasma"
        )
        self.ax_spec.set_title("Spectrogram", color="#00FFFF", fontsize=11, weight="bold")
        self.ax_spec.set_xlabel("Time (s)", color="white")
        self.ax_spec.set_ylabel("Frequency (Hz)", color="white")

        # Colorbar (disesuaikan agar kontras)
        cbar = self.fig.colorbar(pcm, ax=self.ax_spec)
        cbar.set_label("Power/frequency (dB/Hz)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

        self.canvas.draw()

# Main Program
if __name__ == "__main__":
    root = tk.Tk()
    app = DFTSpectrogramGUI(root)
    root.mainloop()
