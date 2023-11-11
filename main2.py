import numpy as np
import matplotlib.pyplot as plt
from main import extract_signal_segment
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftshift, fftfreq
from scipy.signal import windows, spectrogram

from main import task_4


def t1_plot_all_wins():
    window_functions = {
        "Modified Bartlett-Hann": windows.barthann,
        "Bartlett": windows.bartlett,
        "Blackman": windows.blackman,
        "Minimum 4-term Blackman-Harris": windows.blackmanharris,
        "Bohman": windows.bohman,
        "Rectangular": windows.boxcar,
        "Simple Cosine": windows.cosine,
        "Poisson": windows.exponential,
        "Flat Top": windows.flattop,
        "Hamming": windows.hamming,
        "Hann": windows.hann,
        "Parzen": windows.parzen,
        "Triangular": windows.triang,
        "Tukey": windows.tukey
    }

    plt.figure(figsize=(15, 10))
    for name, window in window_functions.items():
        plt.plot(window(51), label=name)

    plt.title('Various Window Functions')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2_task_1_all.png")
    plt.show()

def plot_window_and_spectrum(window_name, window_length, wnum=0):
    signal = windows.get_window(window_name, window_length)
    freqs = fftshift(fftfreq(len(signal)))
    amplitude_spectrum = np.abs(fftshift(np.fft.fft(signal)))

    plt.subplot(2, 2, 1 + wnum*2)
    plt.plot(signal)
    plt.title(f'Time Domain: {window_name} Window')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2 + wnum*2)
    plt.semilogy(freqs, amplitude_spectrum)
    plt.title(f'Frequency Domain: {window_name} Window')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

def task_1():
    t1_plot_all_wins()

    plt.figure(figsize=(12, 5))
    plot_window_and_spectrum('bartlett', 256, wnum=0)
    plot_window_and_spectrum('blackman', 256, wnum=1)
    plt.savefig("2_task_1_bartlett_blackman.png")
    plt.tight_layout()
    plt.show()


def task_2():
    sample_rate, duration = 128, 1
    time_s = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    frequencies = [2, 2.5]

    plt.figure(figsize=(15, 10))
    n_cols = 3

    for i, freq in enumerate(frequencies, start=1):
        signal = np.sin(2 * np.pi * freq * time_s)

        plt.subplot(len(frequencies), n_cols, n_cols*i-2)
        plt.plot(time_s, signal)
        plt.title(f'Initial {freq} Hz Sine')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        fft_freqs = fftshift(fftfreq(time_s.shape[0], d=1/sample_rate))
        signal_fft = fftshift(fft(signal)) / time_s.shape[0]
        signal_amps = np.abs(signal_fft)

        plt.subplot(len(frequencies), n_cols, n_cols*i-1)
        plt.plot(fft_freqs, signal_amps)
        plt.title(f'Amplitude Spectrum of {freq} Hz Sine (No Window)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(-10, 10)

        bartlett_window = windows.bartlett(time_s.shape[0])
        windowed_signal = signal * bartlett_window
        windowed_signal_fft = fftshift(fft(windowed_signal)) / time_s.shape[0]
        windowed_signal_amps = np.abs(windowed_signal_fft)

        plt.subplot(len(frequencies), n_cols, n_cols*i)
        plt.plot(fft_freqs, windowed_signal_amps)
        plt.title(f'Amplitude Spectrum of {freq} Hz Sine (Bartlett Window)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(-10, 10)

    plt.tight_layout()
    plt.savefig("2_task_2.png")
    plt.show()


def task_3():
    sample_rate, duration = 128, 15
    time_s = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    sin_40 = np.sin(2 * np.pi * 40 * time_s)
    rect_pulse = np.zeros_like(time_s)
    rect_pulse[int(10 * sample_rate):int(11 * sample_rate)] = 1
    random_signal = np.random.rand(len(time_s))
    combined_signal = sin_40 + rect_pulse + random_signal
    
    samples_200ms = int(0.2 * sample_rate)  
    bartlett_window = windows.bartlett(samples_200ms)

    # rect_pulse_2 = np.copy(rect_pulse)
    # for i in range(len(rect_pulse) // samples_200ms):
    #     rect_pulse_2[i*samples_200ms:(i+1)*samples_200ms] *= bartlett_window
    # Sxx = []
    # fs = []
    # ts = []
    # for i in range(len(rect_pulse) // samples_200ms):
    #     t = time_s[i*samples_200ms:(i+1)*samples_200ms]
    #     fs = fftshift(fftfreq(t.shape[0], d=1/sample_rate))
    #     ts.append(t[len(t)//2])
    #     Sxx.append(np.abs(fftshift(fft(rect_pulse_2[i*samples_200ms:(i+1)*samples_200ms])) / t.shape[0]))
    # Sxx = np.copy(np.array(Sxx).T)
    # ts = np.array(ts)
    # print(fs.shape, ts.shape, Sxx.shape)
    # plt.pcolormesh(ts, fs, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title(f'Spectrogram')
    # plt.colorbar(label='Amplitude')
    # plt.show()

    for i, (label, sin_40) in enumerate([
        ('3.1 Sine Wave 40Hz', sin_40),
        ('3.2 Rectangular Pulse', rect_pulse),
        ('3.3 Random Signal', random_signal),
        ('3.4 Combined Signal', combined_signal)
    ], start=1):
        freqs, time_s, Sxx = spectrogram(
            sin_40,
            fs=sample_rate,
            window=bartlett_window,
            noverlap=0,
            nperseg=samples_200ms,
            detrend=False,
            return_onesided=True
        )
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(time_s, freqs, Sxx, shading='nearest')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Spectrogram of {label}')
        plt.colorbar(label='Power spectral density [V^2/Hz]')
        plt.savefig(f"2_task_3_{i}.png")
        plt.show()


def task_4():
    sample_rate, duration_s, signal_amp = 256, 10, 1
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    s_1 = signal_amp * np.sin(2 * np.pi * 10 * time_s)
    s_2 = signal_amp * np.sin(2 * np.pi * 100 * time_s)
    s_sum = s_1 + s_2
    s_12 = np.concatenate((2*s_1, 2*s_2))
    s_21 = np.concatenate((2*s_2, 2*s_1))

    window_lengths = [0.1, 2, 1]
    overlaps = [0, 0, 0.5]

    for i, (window_length, overlap) in enumerate(zip(window_lengths, overlaps), 1):
        for j, (signal, sig_name) in enumerate(zip([s_sum, s_12, s_21], ["sum", "1, 2", "2, 1"]), 1):
            nperseg = int(window_length * sample_rate)
            noverlap = int(overlap * nperseg)
            # bartlett_window = windows.bartlett(nperseg)
            freqs, f_time_s, Sxx = spectrogram(
                signal,
                fs=sample_rate,
                # window=bartlett_window,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=False,
                return_onesided=True,
            )

            plt.figure(figsize=(10, 4))
            plt.pcolormesh(f_time_s, freqs, Sxx, shading='nearest')
            plt.colorbar(label='Power spectral density [V^2/Hz]')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title(f'Spectrogram of Signal {sig_name} (Window: {window_length}s, Overlap: {overlap*100}%)')
            plt.savefig(f"2_task_4_{i}_{j}.png")
            plt.show()

            if i <= 2 and j == 1:
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')
                T, F = np.meshgrid(f_time_s, freqs)
                ax.plot_surface(T, F, 10 * np.log10(Sxx), cmap='viridis')
                ax.set_ylabel('Frequency [Hz]')
                ax.set_xlabel('Time [sec]')
                ax.set_zlabel('Intensity [dB]')
                plt.title(f'3D Spectrogram of Signal {sig_name} (Window: {window_length}s)')
                plt.savefig(f"2_task_4_3D_{i}_{j}.png")
                plt.show()


def task_5():
    sample_rate, duration_s, signal_amp = 128, 3, 1
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    signal = signal_amp * np.sin(2 * np.pi * 20 * time_s)
    break_len = 10

    nperseg = int(0.05 * sample_rate)
    noverlap = nperseg - 1

    for i, bp in enumerate((1.05, 2), start=1):
        break_point = int(bp * sample_rate)
        signal_with_gap = np.copy(signal)
        signal_with_gap[break_point:break_point+break_len] = 0
        
        freqs, f_time_s, Sxx = spectrogram(
            signal_with_gap,
            fs=sample_rate,
            # window=bartlett_window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            return_onesided=True,
        )

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(f_time_s, freqs, Sxx, shading='nearest')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Spectrogram with Gap (Break at {bp} sec)')
        plt.colorbar(label='Power spectral density [V^2/Hz]')
        plt.savefig(f"2_task_5_{i}.png")
        plt.show()


def plot_avg_spectral_density_change(signal, t1, t2, f1, f2, sample_rate, nperseg=None, noverlap=None, detrend='constant'):
    time_s, extracted_signal = extract_signal_segment(signal, t1, t2, sample_rate)

    f, t, Sxx = spectrogram(
        extracted_signal,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        return_onesided=True
    )

    t += time_s[0]

    f_idx = np.where((f >= f1) & (f <= f2))

    avg_spectral_density = np.mean(Sxx[f_idx], axis=0)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(time_s, extracted_signal)
    plt.title('Extracted Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(t, avg_spectral_density)
    plt.title('Average Spectral Density Change')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Spectral Density')
    plt.grid()

def task_6():
    sample_rate, duration_s, signal_amp = 256, 20, 1
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    # signal = signal_amp * np.sin(2 * np.pi * 1 * time_s)

    # initial_freq = 0.5
    # final_freq = 3
    # freq_change = (final_freq - initial_freq) / duration_s
    # signal = np.sin(2 * np.pi * (initial_freq + freq_change/2 * time_s) * time_s)

    max_amplitude = 1
    amplitude_change = max_amplitude / duration_s
    signal = np.sin(2 * np.pi * 1 * time_s) * (amplitude_change * time_s)


    plot_avg_spectral_density_change(signal, 2, 8, 10, 30, sample_rate, nperseg=25, noverlap=20, detrend=False)
    plt.savefig("2_task_6.png")
    plt.show()

    plot_avg_spectral_density_change(signal, 2, 8, 10, 30, sample_rate, nperseg=25, noverlap=20)
    plt.savefig("2_task_6_detrend.png")
    plt.show()


def main():
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()


if __name__ == "__main__":
    main()
