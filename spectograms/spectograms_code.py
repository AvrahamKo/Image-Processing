import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.io.wavfile import write
from scipy.io import wavfile

################################ TASK1 ###############################
def create_watermark(audio, sr,frequency,strength):
    duration = len(audio) / sr
    t = np.linspace(0, duration, len(audio))
    watermark =np.sin(2*np.pi*frequency*t)*strength
    return audio + watermark

def show_spectrogram(audio,name):
    # Plot the spectrogram
    stft = librosa.stft(audio, n_fft=1024, hop_length=256)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Plot the spectrogram on a linear scale
    plt.figure(figsize=(24, 12))
    librosa.display.specshow(
        stft_db, sr=sr, hop_length=256, x_axis="time", y_axis="linear"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram (Linear Scale)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

def save_audio(filename, signal, sr):
    write(filename, sr, (signal * 32767).astype(np.int16))

############################## TASK2 #################################
def plot_spectrograms(folder_path):
    files = list(Path(folder_path).glob('*.wav'))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for idx, file in enumerate(files):
        # Load audio
        sample_rate, data = wavfile.read(file)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Create spectrogram
        row, col = idx // 3, idx % 3
        axes[row, col].specgram(data, Fs=sample_rate, NFFT=2048, noverlap=1024)
        axes[row, col].set_ylim(0, 22000)  # Focus on relevant frequency range
        axes[row, col].set_title(f'File: {file.name}')
        axes[row, col].set_xlabel('Time (s)')
        axes[row, col].set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()



def analyze_watermark(file_path, freq_range=(18000, 20000)):
    """
        Detect and characterize a static watermark in a WAV file.

        Parameters
        ----------
        file_path : str
            Path to the input WAV file.
        freq_range : tuple of (float, float), optional
            Frequency band (min_freq, max_freq) in Hz to isolate watermark.

        Returns
        -------
        float
            Dominant watermark frequency in Hz extracted from the pattern's FFT.
        np.ndarray
            Average spectral energy over the watermark band for each time frame.
        """
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    f, t, Sxx = signal.spectrogram(data, sample_rate, nperseg=2048, noverlap=1024)

    # Focus on target frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    watermark_band = Sxx[freq_mask]

    # Extract pattern characteristics
    pattern = np.mean(watermark_band, axis=0)

    # FFT of the pattern to find main frequency
    pattern_fft = np.abs(np.fft.fft(pattern))
    pattern_freqs = np.fft.fftfreq(len(pattern), t[1] - t[0])

    # Get dominant frequency
    main_freq = pattern_freqs[np.argmax(pattern_fft[1:]) + 1]

    return abs(main_freq), pattern

def add_modulated_watermark(audio_path,
                                center_freq=18000,
                                mod_freq=0.5,  # Modulation frequency in Hz
                                freq_dev=2000,  # Frequency deviation in Hz
                                amplitude=0.1):
        """
        Add frequency-modulated watermark to audio.
        """
        sr, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

        t = np.arange(len(audio)) / sr

        # Create frequency modulation
        instant_freq = center_freq + freq_dev * np.sin(2 * np.pi * mod_freq * t)

        # Generate phase by integrating frequency
        phase = 2 * np.pi * np.cumsum(instant_freq) / sr

        # Create watermark signal
        watermark = amplitude * np.sin(phase)

        watermarked = audio + watermark
        show_spectrogram(watermarked,sr)

############################ TASK3 ###################################
def analyze_highest_frequency(file_path):
    """
    Analyzes the highest frequency of a WAV file using FFT.

    Parameters:
        file_path (str): Path to the WAV file.

    Returns:
        float: Highest frequency (Hz) with non-negligible magnitude.
    """
    # Read the WAV file
    sample_rate, data = wavfile.read(file_path)

    # Convert to mono if the file is stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalize the data
    data = data / np.max(np.abs(data))

    # Perform FFT
    N = len(data)
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)
    highest_frequency = freqs[-1] if len(freqs) > 0 else 0
    return highest_frequency

def calculate_speedup_ratio(file1_path, file2_path):
    """
    Calculates the speedup ratio between two WAV files using the highest frequency.

    Parameters:
        file1_path (str): Path to the first WAV file.
        file2_path (str): Path to the second WAV file.

    Returns:
        dict: A dictionary containing the highest frequencies and speedup ratio.
    """
    freq1 = analyze_highest_frequency(file1_path)
    freq2 = analyze_highest_frequency(file2_path)

    # Determine speedup ratio
    speedup_ratio = freq2 / freq1 if freq1 < freq2 else freq1 / freq2

    # Determine the method for each file
    method1 = "Frequency domain slowdown" if freq2 < freq1 else "Time domain slowdown"
    method2 = "Frequency domain slowdown" if freq1 < freq2 else "Time domain slowdown"

    return {
        "File 1": {"Path": file1_path, "Highest Frequency (Hz)": freq1, "slowdown Method": method1},
        "File 2": {"Path": file2_path, "Highest Frequency (Hz)": freq2, "slowdown Method": method2},
        "slowdown Ratio (x)": speedup_ratio
    }


if __name__ == "__main__":
    ###FOR TASK 1 - Good and bad watermark
    sr = 44100
    audio_path = "Task 1/task1.wav"
    audio, _ = librosa.load(audio_path, sr=sr)
    # Create good and bad watermar
    good_watermarked = create_watermark(audio, sr,frequency = 20000,strength= 0.001)
    bad_watermarked = create_watermark(audio, sr,frequency = 10000,strength= 0.1)

    # Plot spectrograms
    show_spectrogram(audio,"real")
    show_spectrogram(bad_watermarked,"bed")
    show_spectrogram(good_watermarked,"good")

    #save audio
    save_audio("Task 1/good_watermarked.wav", good_watermarked, sr)
    save_audio("Task 1/bad_watermarked.wav", bad_watermarked, sr)


    ###FOR TASK 2 - Classify
    # Process files
    patterns = {}
    folder_path = 'Task 2'
    plot_spectrograms(folder_path)
    files = list(Path(folder_path).glob('*.wav'))

    for file in files:
        freq, pattern = analyze_watermark(str(file))
        patterns[file.name] = (freq, pattern)

    # Group similar frequencies
    groups = {}
    for file, (freq, pattern) in patterns.items():
        rounded_freq = round(freq, 2)
        # rounded_freq = freq
        if rounded_freq not in groups:
            groups[rounded_freq] = []
        groups[rounded_freq].append(file)

    # Print results and plot patterns
    # plot_patterns_enhanced(patterns)
    for freq, files in groups.items():
        print(f"\nWatermark frequency: {freq} Hz")
        print("Files:", ", ".join(files))

    audio_path = "Task 2/3_watermarked.wav"
    add_modulated_watermark(audio_path)

    ###FOR TASK 3 - Speed VS. frequency

    # Calculate the speedup ratio
    results = calculate_speedup_ratio("Task 3/task3_watermarked_method1.wav", "Task 3/task3_watermarked_method2.wav")

    # Print results
    print("\nAnalysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")