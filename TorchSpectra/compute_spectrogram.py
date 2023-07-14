import torch

all = [
    "compute_spectrogram",
]

def compute_spectrogram(input_vector, sample_rate, window_size=None, hop_size=None, n_fft=None, window_type="hann"):
    """
    Computes the spectrogram using PyTorch on a GPU, considering the inputs similar to the scipy.signal.periodogram function,
    and also returns the frequency axis.

    Parameters:
    ----------
    input_vector : list or np.array
        The input signal or data sequence for which to compute the spectrogram.
    sample_rate : int
        The sample rate of the input vector, measured in Hertz (Hz).
    window_size : int, optional
        The size of the analysis window used in the FFT computation.
        This parameter determines the size of the analysis window used in the
        FFT computation. It specifies the number of samples in each window. A larger
        window size provides better frequency resolution but sacrifices time resolution.
        It is usually a power of 2 for efficient FFT computation.
    hop_size : int, optional
        The number of samples between the start of one window and the start
        of the next window in the spectrogram.
        A smaller hop size leads to a higher overlap, while a larger hop size
        reduces the overlap. Increasing the overlap can improve time resolution
        by reducing the spacing between consecutive time frames in the spectrogram.
        However,it also increases computational complexity. A smaller hop size
        increases the time resolution but may introduce more spectral leakage.
    n_fft : int, optional
        The number of FFT points or bins used in the FFT computation.
        The parameter specifies the number of FFT points or bins used in the FFT
        computation. It determines the frequency resolution of the resulting spectrogram.
        More FFT points yield finer frequency resolution but increase computational complexity.
        Typically, n_fft is also a power of 2 for efficient FFT computation.
    window_type : str, optional
        The type of window function to be used. Available options: "hann" (default), "hamming",
        "boxcar", "bartlett", "blackman", "kaiser".


    Returns:
    -------
    power_spectrogram : np.array
        The power spectrogram computed using the STFT.
    time_axis : np.array
        The time axis corresponding to the spectrogram. It provides the time points at which the spectrogram is computed.
    frequency_axis : np.array
        The frequency axis corresponding to the spectrogram. It provides the frequencies at which the power spectrum is
        calculated.

    Example usage I (General):
    --------------
    >>> input_vector = [...]  # Your input vector
    >>> sample_rate = 44100  # Sample rate of the input vector
    >>> window_size = 1024  # Size of the analysis window (in samples)
    >>> hop_size = 512  # Hop size (in samples)
    >>> n_fft = 1024  # Number of FFT bins
    >>> time_axis, frequency_axis, power_spectrogram = compute_spectrogram(input_vector, sample_rate, window_size, hop_size, n_fft)

    Example usage II (Synthetic signal):
    --------------
    >>>  # Generate a sample signal
    >>>  fs = 2  # Sample rate (Hz)
    >>>  t = np.arange(0, 180, 1/fs)  # Time vector
    >>>  f1 = 0.8  # Frequency of the signal
    >>>  sig = np.sin(2 * np.pi * f1 * t)
    >>>  time_axis, frequency_axis, power_spectrogram = compute_spectrogram(input_vector=sig, sample_rate=2, window_type="hann")
    >>>
    >>>  # Plot the spectrogram
    >>>  plt.plot(frequency_axis, power_spectrogram)
    >>>  plt.xlabel('Frequency')
    >>>  plt.ylabel('Power Spectrum')
    >>>  plt.show()

    Example usage III (Synthetic signal):
    --------------
    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> mod = 500*np.cos(2*np.pi*0.25*time)
    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    >>> rng = np.random.default_rng()
    >>> noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> noise *= np.exp(-time/5)
    >>> x = carrier + noise
    >>>
    >>> # frequency_axis, power_spectrogram = compute_spectrogram(input_vector=sig, sample_rate=2, window_type="hann")
    >>> time_axis, frequency_axis, power_spectrogram = compute_spectrogram(input_vector=x, sample_rate=fs, window_size=3*60, hop_size=2*60, n_fft=3*60, window_type="hann")
    >>>
    >>> # Plot the computed spectrogram
    >>> fig, ax = plt.subplots(figsize=(10,6))
    >>> plt.pcolormesh(time_axis, frequency_axis, power_spectrogram, shading='gouraud', cmap="turbo")
    >>> ax.tick_params(axis='x', rotation=45, labelsize=13)
    >>> ax.tick_params(axis='y', rotation=45, labelsize=13)
    >>> ax.set_xlabel("Time [sec]", fontsize=15)
    >>> ax.set_ylabel("Frequency [Hz]", fontsize=15)
    >>> ax.grid(False)
    >>> plt.show()
    """

    def boxcar_window(window_size):
        # Create a tensor of ones with the desired window size
        window = torch.ones(window_size)
        return window

    # Handle input variables
    if window_size == None:
        # If None the length of x will be used.
        window_size = len(input_vector)
    if hop_size == None:
        # If None the length of x will be used.
        hop_size = len(input_vector)
    if n_fft is None:
        # If None the length of x will be used.
        n_fft =  len(input_vector)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input vector to a PyTorch tensor and move it to the GPU
    input_tensor = torch.tensor(input_vector, dtype=torch.float64).to(device)

    # Print the device and tensor contents
    if device.type == "cuda":
        print("INFO: GPU available. Tensor moved to GPU.")
    else:
        print("INFO: No GPU available. Tensor remains on CPU.")

    # Compute the STFT (Short-Time Fourier Transform)
    if window_type == "boxcar":
        window = boxcar_window(window_size)
    else:
        window = getattr(torch, f"{window_type}_window")(window_size)
    spectrogram = torch.stft(input_tensor, n_fft=n_fft, hop_length=hop_size, win_length=window_size,
                             window=window.cuda(), center=False, return_complex=True, onesided=True,
                             normalized=True)

    # Convert complex spectrogram to magnitude spectrogram
    magnitude_spectrogram = torch.abs(spectrogram)

    # Compute the power spectrogram
    # power_spectrogram = magnitude_spectrogram.pow(2)
    power_spectrogram = magnitude_spectrogram ** 2

    # Scale the spectrogram by the window energy
    # power_spectrogram *= window_size / float(hop_size * sample_rate)

    # Compute the time axis
    total_time = len(input_vector) / sample_rate
    num_segments = power_spectrogram.shape[1]
    if num_segments == 1:
        time_axis = torch.zeros(1)
    else:
        time_step = total_time / (num_segments - 1)
        time_axis = torch.arange(0, total_time + time_step, time_step)

    # Compute the frequency axis
    frequency_axis = torch.linspace(0, sample_rate/2, n_fft//2 + 1)

    # Convert output vector to a Numpy array and move it to the CPU
    if device.type == "cuda":
        power_spectrogram = power_spectrogram.cpu().detach().numpy()
    time_axis = time_axis.detach().numpy()
    frequency_axis = frequency_axis.detach().numpy()

    return time_axis, frequency_axis, power_spectrogram
