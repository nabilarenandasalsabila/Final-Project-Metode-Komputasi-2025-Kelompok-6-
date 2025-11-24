import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Backend_DFT import analyze_audio, compute_fft
from scipy.signal import chirp
import sounddevice as sd
from scipy.io.wavfile import write

class DFTSpectrogramGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 DFT + Spectrogram Analyzer (Interaktif 3D)")
        self.root.geometry("400x500")
        self.root.configure(bg="#1E1E2F")

        button_frame = tk.Frame(root, bg="#2C2C3E", bd=2, relief="ridge")
        button_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        title_label = tk.Label(
            button_frame, text="🎧 AUDIO TOOLKIT", bg="#2C2C3E",
            fg="white", font=("Century Gothic", 14, "bold")
        )
        title_label.pack(pady=(10, 25))

        def create_button(text, color, command):
            btn = tk.Button(
                button_frame, text=text, bg=color, fg="white",
                activebackground="#57CC99", activeforeground="black",
                font=("Segoe UI", 10, "bold"), width=25, height=2,
                relief="flat", cursor="hand2", command=command
            )
            btn.pack(pady=8)
            return btn

        create_button("🎼 Load WAV File", "#6A5ACD", self.load_wav)
        create_button("🎹 Generate Test Signal", "#4682B4", self.generate_signal)
        create_button("📈 Show Amplitude Spectrum", "#00CED1", self.show_amplitude_plot)
        create_button("📐 Show Phase Spectrum", "#BA55D3", self.show_phase_plot)
        create_button("🌈 Show Spectrogram", "#FF6347", self.show_spectrogram_plot)
        create_button("🚪 Exit", "#C94C4C", self.root.quit)

        self.result = None

    def load_wav(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if not file_path:
            return
        try:
            self.result = analyze_audio(file_path)
            messagebox.showinfo("Sukses", "File berhasil dimuat. Silakan pilih jenis grafik yang ingin ditampilkan.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca file:\n{e}")

    def generate_signal(self):
        fs = 44100
        t = np.linspace(0, 5, fs * 5)
        signal = chirp(t, f0=200, f1=10000, t1=5, method='linear')
        write("test_signal.wav", fs, (signal * 32767).astype(np.int16))
        sd.play(signal, fs)
        self.result = analyze_audio("test_signal.wav")
        messagebox.showinfo("Sinyal Uji", "Sinyal chirp berhasil dibuat dan dimainkan.\nSilakan pilih jenis grafik yang ingin ditampilkan.")

    def show_amplitude_plot(self):
        if not self.result:
            messagebox.showwarning("Belum Ada Data", "Silakan load file WAV atau generate sinyal terlebih dahulu.")
            return

        signal = self.result["signal"]
        sr = self.result["sample_rate"]
        fft_result = compute_fft(signal)
        N = len(fft_result)
        freq = np.fft.fftfreq(N, 1 / sr)

        fig = plt.figure(figsize=(8, 6), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        x = freq[:N // 2]
        y = np.zeros_like(x)
        z = np.abs(fft_result[:N // 2])
        ax.plot(x, y, z, color="#00CED1")
        ax.set_title("DFT Amplitude Spectrum", color="black", fontsize=12, weight="bold")
        ax.set_xlabel("Frequency (Hz)", color="black")
        ax.set_ylabel("Reference Axis", color="black")
        ax.set_zlabel("Amplitude", color="black")
        ax.auto_scale_xyz(x, y, z)
        plt.tight_layout()
        plt.show()

    def show_phase_plot(self):
        if not self.result:
            messagebox.showwarning("Belum Ada Data", "Silakan load file WAV atau generate sinyal terlebih dahulu.")
            return

        signal = self.result["signal"]
        sr = self.result["sample_rate"]
        fft_result = compute_fft(signal)
        N = len(fft_result)
        freq = np.fft.fftfreq(N, 1 / sr)

        fig = plt.figure(figsize=(8, 6), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        x = freq[:N // 2]
        y = np.zeros_like(x)
        z_phase = np.angle(fft_result[:N // 2])
        ax.plot(x, y, z_phase, color="#BA55D3")
        ax.set_title("DFT Phase Spectrum", color="black", fontsize=12, weight="bold")
        ax.set_xlabel("Frequency (Hz)", color="black")
        ax.set_ylabel("Reference Axis", color="black")
        ax.set_zlabel("Phase (radians)", color="black")
        ax.auto_scale_xyz(x, y, z_phase)
        plt.tight_layout()
        plt.show()

    def show_spectrogram_plot(self):
        if not self.result:
            messagebox.showwarning("Belum Ada Data", "Silakan load file WAV atau generate sinyal terlebih dahulu.")
            return

        f = self.result["frequencies"]
        t = self.result["times"]
        Sxx = self.result["spectrogram"]
        T, F = np.meshgrid(t, f)
        Z = 20 * np.log10(Sxx + 1e-6)

        fig = plt.figure(figsize=(8, 6), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        surf = ax.plot_surface(T, F, Z, cmap="plasma", edgecolor='none')
        ax.set_title("Spectrogram", color="black", fontsize=12, weight="bold")
        ax.set_xlabel("Time (s)", color="black")
        ax.set_ylabel("Frequency (Hz)", color="black")
        ax.set_zlabel("Power (dB)", color="black")
        ax.auto_scale_xyz(t, f, Z)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.show()

# Main Program
if __name__ == "__main__":
    root = tk.Tk()
    app = DFTSpectrogramGUI(root)
    root.mainloop()
    