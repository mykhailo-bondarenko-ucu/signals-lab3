import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import square, welch
from scipy.fft import fft, fftshift, fftfreq, ifft

def spectral_leakage_c(c_freq, signal_freq, signal_amp, time_s):
    # c_freq = m/(N * T_s)
    # signal_freq = omega_star / T_s
    # time_s = np.arange(N) * T_s
    return (signal_amp / (time_s.shape[0] * 2j)) * np.sum(np.exp(
        2j * np.pi * (signal_freq - c_freq) * time_s
    ) - np.exp(
        2j * np.pi * (-signal_freq - c_freq) * time_s
    ))

def task_1():
    sample_rate, duration_s, signal_amp = 128, 1, 1
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    freqs = [2, 2.5, 40, 100, 600]

    plt.figure(figsize=(12, 10))
    for i, signal_freq in enumerate(freqs, start=1):
        signal = signal_amp * np.sin(2 * np.pi * signal_freq * time_s)
        fft_amps = np.abs(fftshift(fft(signal)) / time_s.shape[0])
        fft_freqs = fftshift(fftfreq(time_s.shape[0], d=1/sample_rate))
        fft_freqs, fft_amps = fft_freqs, fft_amps
        lk_amps = np.abs(np.array([
            spectral_leakage_c(c_freq, signal_freq, signal_amp, time_s)
            for c_freq in fft_freqs
        ]))
        plt.subplot(len(freqs), 2, 2*i-1)
        plt.plot(time_s, signal)
        plt.title(f"Time Domain: {signal_freq} Hz")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(len(freqs), 2, 2*i)
        plt.stem(fft_freqs, fft_amps, use_line_collection=True, label="Actual frequencies")
        plt.plot(fft_freqs, lk_amps, 'r-', label='Predicted Leakage')
        plt.legend()
        plt.title(f"Frequency Domain: {signal_freq} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig("task_1.png")
    plt.show()

def task_2():
    sample_rate, duration_s, signal_amp = 256, 10, 1
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    s_1 = signal_amp * np.sin(2 * np.pi * 10 * time_s)
    s_2 = signal_amp * np.sin(2 * np.pi * 100 * time_s)
    s_sum = s_1 + s_2
    s_12 = np.concatenate((2*s_1, 2*s_2))
    s_21 = np.concatenate((2*s_2, 2*s_1))
    plt.figure(figsize=(16, 8))
    amps = []
    for i, (signal, name) in enumerate([(s_sum, "Signal sum"), (s_12, "Signal 1, 2"), (s_21, "Signal 2, 1")], start=1):
        time_s = np.arange(len(signal)) / sample_rate
        fft_amps = np.abs(fftshift(fft(signal)) / time_s.shape[0])
        fft_freqs = fftshift(fftfreq(time_s.shape[0], d=1/sample_rate))
        fft_freqs, fft_amps = fft_freqs, fft_amps
        amps.append(fft_amps)
        plt.subplot(3, 2, 2*i-1)
        plt.plot(time_s, signal)
        plt.title(f"Time Domain: {name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.subplot(3, 2, 2*i)
        plt.stem(fft_freqs, fft_amps, use_line_collection=True)
        plt.title(f"Frequency Domain: {name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
    print(f"{np.allclose(amps[1], amps[2]) = }")
    
    plt.tight_layout()
    plt.savefig("task_2.png")
    plt.show()


def task_3():
    sample_rate, duration_s, signal_amp = 128, 3, 1
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    signal = signal_amp * np.sin(2 * np.pi * 20 * time_s)
    break_len = 10

    plt.figure(figsize=(16, 8))
    for i, bp in enumerate((1.05, 2), start=1):
        break_point = int(bp * sample_rate)
        signal_with_gap = np.copy(signal)
        signal_with_gap[break_point:break_point+break_len] = 0

        fft_freqs = fftshift(fftfreq(time_s.shape[0], d=1/sample_rate))

        sig_w_gap_fft = fftshift(fft(signal_with_gap)) / time_s.shape[0]
        gap_fft_amps = np.abs(sig_w_gap_fft)
        gap_fft_angs = np.angle(sig_w_gap_fft)

        # detect main frequency (two of them mirrored)
        main_freq_indx = np.where(gap_fft_amps > 0.4)
        assert len(main_freq_indx[0]) == 2

        # form new spectrum
        sig_w_gap_fft[main_freq_indx] = 0
        delta_sig = ifft(fftshift(sig_w_gap_fft * time_s.shape[0])).real

        m = delta_sig * signal
        win_conv = np.sum([m[i:len(m)-break_len+i] for i in range(break_len)], axis=0)
        pos = np.argmin(win_conv)

        plt.subplot(2, 3, 3*i-2)
        plt.plot(time_s, signal_with_gap)
        plt.title(f"Time Domain: break at {bp}s")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.subplot(2, 3, 3*i-1)
        plt.plot(fft_freqs, gap_fft_amps, 'r', label="Amplitude")
        plt.plot(fft_freqs, gap_fft_angs, 'b', label="Phase")
        plt.legend()
        plt.title(f"Frequency Domain: break at {bp}s")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")

        plt.subplot(2, 3, 3*i)
        plt.plot(time_s, delta_sig)
        plt.axvspan(time_s[pos], time_s[pos+break_len+1], color='orange', alpha=0.7, label=f'Detected break region ({time_s[pos]}s)')
        plt.legend()
        plt.title(f"Time domain after gapless signal substraction")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.savefig("task_3.png")
    plt.show()


def task_4():
    sample_rate, duration_s, signal_amp = 512, 3, 1
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)

    plt.figure(figsize=(16, 8))
    for i, freq in enumerate((10, 100), start=1):
        signal = signal_amp * square(2 * np.pi * freq * time_s)

        fft_freqs = fftshift(fftfreq(time_s.shape[0], d=1/sample_rate))
        signal_fft = fftshift(fft(signal)) / time_s.shape[0]
        signal_amps = np.abs(signal_fft)

        plt.subplot(2, 3, 3*i-2)
        plt.plot(time_s, signal)
        plt.title(f'Time Domain: {freq} Hz Rectangular Pulse')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(2, 3, 3*i-1)
        plt.plot(fft_freqs, signal_amps)
        plt.title(f'Frequency Domain: {freq} Hz Rectangular Pulse')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.subplot(2, 3, 3*i)
        plt.plot(fft_freqs, signal_amps)
        plt.title(f'Frequency Domain: {freq} Hz Rectangular Pulse ([-40Hz, 40Hz])')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(-40, 40)

    plt.tight_layout()
    plt.savefig("task_4.png")
    plt.show()


def task_5():
    sample_rate, duration_s, signal_amp = 512, 30, 1
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)

    pulse_durations = [0.1, 1, 10]
    time_shifts = [0, 5]

    for do_lim in [False, True]:
        plt.figure(figsize=(24, 8))
        for i, pulse_duration in enumerate(pulse_durations, start=1):
            for j, time_shift in enumerate(time_shifts, start=1):
                signal = np.zeros_like(time_s)
                start_idx = int(time_shift * sample_rate)
                end_idx = int((time_shift + pulse_duration) * sample_rate)
                signal[start_idx:end_idx] = signal_amp

                fft_freqs = fftshift(fftfreq(time_s.shape[0], d=1/sample_rate))
                signal_fft = fftshift(fft(signal)) / time_s.shape[0]
                signal_amps = np.abs(signal_fft)
                signal_phase = np.angle(signal_fft)

                idx = (i - 1) * len(time_shifts) + j
                plt.subplot(len(pulse_durations), len(time_shifts)*3, 3*idx-2)
                plt.plot(time_s, signal)
                plt.title(f'Time Domain: d:{pulse_duration}s, s:{time_shift}s')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')

                plt.subplot(len(pulse_durations), len(time_shifts)*3, 3*idx-1)
                plt.plot(fft_freqs, signal_amps)
                plt.title(f'Amplitude Spectrum: d:{pulse_duration}s, s:{time_shift}s')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude')
                if do_lim: plt.xlim(-3, 3)

                plt.subplot(len(pulse_durations), len(time_shifts)*3, 3*idx)
                plt.plot(fft_freqs, signal_phase)
                plt.title(f'Phase Spectrum: d:{pulse_duration}s, s:{time_shift}s')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Phase (radians)')
                if do_lim: plt.xlim(-3, 3)

        plt.tight_layout()
        plt.savefig(f"task_5_{'lim' if do_lim else 'no_lim'}.png")
        plt.show()


def task_6():
    sample_rate, duration_s = 1000, 10
    time_s = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    
    signal = np.random.rand(len(time_s))

    fft_freqs = fftshift(fftfreq(time_s.shape[0], d=1/sample_rate))
    signal_fft = fftshift(fft(signal)) / time_s.shape[0]
    signal_amps = np.abs(signal_fft)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_s, signal)
    plt.title('Random Signal in Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(fft_freqs, signal_amps)
    plt.title('Amplitude Spectrum of Random Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(f"task_6.png")
    plt.show()


def t_7_plot(amp, duration, sample_rate, mean_noise, freq, i, n_plots):
    time_s = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = amp * np.sin(2 * np.pi * freq * time_s) + np.random.normal(mean_noise, scale=1, size=time_s.shape)
    
    fft_freqs = fftshift(fftfreq(time_s.shape[0], d=1/sample_rate))
    signal_fft = fftshift(fft(signal)) / time_s.shape[0]
    signal_amps = np.abs(signal_fft)

    plt.subplot(n_plots, 2, 2*i-1)
    plt.plot(time_s, signal)
    plt.title(f'Time Domain: d: {duration}s, fs: {sample_rate}Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')

    plt.subplot(n_plots, 2, 2*i)
    plt.plot(fft_freqs, signal_amps)
    plt.title(f'Amplitude Spectrum: d: {duration}s, fs: {sample_rate}Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (V)')

def task_7():
    freq, amp, mean_noise = 20, 1, 10

    durations = [0.5, 1, 10, 100, 1000]
    plt.figure(figsize=(16, len(durations)*4))
    sample_rate = 128
    for i, duration in enumerate(durations, start=1):
        t_7_plot(amp, duration, sample_rate, mean_noise, freq, i, len(durations))
    plt.tight_layout()
    plt.savefig("task_7_durations.png")
    plt.show()

    # 7.2
    sample_rates = [128, 1280, 12800, 128000]
    plt.figure(figsize=(16, len(sample_rates)*4))
    for i, sample_rate in enumerate(sample_rates, start=1):
        duration = 0.5  # fixed duration
        t_7_plot(amp, duration, sample_rate, mean_noise, freq, i, len(sample_rates))
        plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig("task_7_sample_rates.png")
    plt.show()


def task_8():
    freq, amp, duration, sample_rate = 20.5, 1, 1, 1000
    zero_paddings = [0, 10, 100, 1000, 10000]
    
    plt.figure(figsize=(16, len(zero_paddings)*4))
    for i, n_zeros in enumerate(zero_paddings, start=1):
        time_s = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        signal = amp * np.sin(2 * np.pi * freq * time_s)
        signal = np.concatenate((signal, np.zeros(n_zeros)))

        fft_freqs = fftshift(fftfreq(signal.shape[0], d=1/sample_rate))
        signal_fft = fftshift(fft(signal)) / signal.shape[0]
        signal_amps = np.abs(signal_fft)

        plt.subplot(len(zero_paddings), 2, 2*i-1)
        plt.plot(np.arange(signal.shape[0]) / sample_rate, signal)
        plt.title(f'Time Domain: padding: {n_zeros}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')

        plt.subplot(len(zero_paddings), 2, 2*i)
        plt.stem(fft_freqs, signal_amps, use_line_collection=True)
        plt.title(f'Amplitude Spectrum: padding: {n_zeros}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (V)')
        plt.xlim(19, 22)

    plt.tight_layout()
    plt.savefig("task_8.png")
    plt.show()


def task_9():
    data, sample_rate = sf.read("./audio_44k.wav")
    time_s = np.linspace(0, len(data) / sample_rate, len(data), endpoint=False)
    transformed_data = fft(data)
    restored_data = ifft(transformed_data).real
    mse = np.std(data - restored_data)

    print(f'Mean Squared Error (MSE): {mse}')

    plt.figure(figsize=(16, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_s[:1000], data[:1000])
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(time_s[:1000], restored_data[:1000])
    plt.title('Restored Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 3)
    plt.plot(time_s[:1000], data[:1000] - restored_data[:1000])
    plt.title('Difference between Original and Restored Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig("task_9.png")
    plt.show()


def plot_psd(filename):
    data, sample_rate = sf.read(filename)
    frequencies, psd = welch(data, sample_rate, nperseg=1024)

    plt.semilogy(frequencies, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (V^2/Hz)')
    plt.title('Power Spectral Density')
    plt.grid()

def task_10():
    filename = "./audio_44k.wav"
    
    plt.figure(figsize=(10, 6))
    plot_psd(filename)

    plt.tight_layout()
    plt.savefig("task_10.png")
    plt.show()


def extract_signal_segment(
        signal: np.ndarray,
        start_time_sec: np.dtype('float64'),
        end_time_sec: np.dtype('float64'),
        sample_rate: np.dtype('int64'),
    ):
    if start_time_sec < 0 or end_time_sec > len(signal) / sample_rate:
        raise ValueError("Start time or end time is outside the time range.")
    
    time_vector = np.linspace(0, len(signal) / sample_rate, len(signal))

    start_index = np.argmax(time_vector >= start_time_sec)
    end_index = np.argmin(time_vector <= end_time_sec)

    while (end_index+1) < len(time_vector) and time_vector[end_index] == time_vector[end_index+1]:
        end_index += 1

    extracted_time = time_vector[start_index:end_index + 1]
    extracted_signal = signal[start_index:end_index + 1]

    return extracted_time, extracted_signal

def plot_segment_spectrum(signal, start_time_sec, end_time_sec, sample_rate):
    time, signal = extract_signal_segment(signal, start_time_sec, end_time_sec, sample_rate)
    signal_fft = fftshift(fft(signal)) / len(signal)
    freqs = fftshift(fftfreq(len(signal), d=1/sample_rate))
    amplitude_spectrum = np.abs(signal_fft)
    phase_spectrum = np.angle(signal_fft)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, signal)
    plt.title('Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 2)
    plt.plot(freqs, amplitude_spectrum)
    plt.title('Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(freqs, phase_spectrum)
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')

def task_11():
    sample_rate = 500
    t1 = np.linspace(0, 1, sample_rate, endpoint=False)
    t2 = np.linspace(1, 2, sample_rate, endpoint=False)
    signal1 = np.sin(2 * np.pi * 10 * t1)
    signal2 = np.sin(2 * np.pi * 20 * t2)
    signal = np.concatenate((signal1, signal2))

    plot_segment_spectrum(signal, 0.8, 1.3, sample_rate)

    plt.tight_layout()
    plt.savefig("task_11.png")
    plt.show()


def main():
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    task_7()
    task_8()
    task_9()
    task_10()
    task_11()

if __name__ == "__main__":
    main()
