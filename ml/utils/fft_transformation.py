import numpy as np
# import pandas as pd
import scipy.signal as sig
from scipy import signal
from scipy.fftpack import fft


def spektrum(y, fs):  # y=signal fs=samplerate
    Y = fft(y)
    T = 1 / fs
    N = Y.size
    f = np.linspace(0.0, fs / 2, N // 2)
    Y_abs = 2.0 / N * np.abs(Y[0:N // 2])
    return f, Y_abs


# %%sampling_frequency is the upper bound of the frequency
sampling_frequency = 64e3
time_step = 1. / sampling_frequency


# sampling_frequency = np.ceil(1 / dt)


def get_spectrum(input_signal, window):
    """
    returns the Fourier power spectrum for a given signal segment
    output is a pandas Series
    output.index is the frequencies
    output.values is the amplitudes for each frequencies
    default moving average window is 10
    """
    input_signal = np.asarray(input_signal, dtype='float64')

    # Remove the mean
    input_signal -= input_signal.mean()

    # Estimate power spectral density using a periodogram.
    frequencies, power_spectrum = signal.periodogram(
        input_signal, sampling_frequency, scaling='spectrum')

    # Run a running windows average of 10-points to smooth the signal (default).
    power_spectrum = pd.Series(power_spectrum, index=frequencies).rolling(window=window).mean()

    return pd.Series(power_spectrum)


def get_segment_spectrum(segment_df, window):
    """
    Get the fourier power spectrum of a given segment.

    Returns the frequencies, and power_spectrum
    """
    _power_spectrum = get_spectrum(segment_df.vibration_data, window=window).dropna()
    # Keep one every 10 samples
    power_spectrum = _power_spectrum.values
    frequencies = _power_spectrum.index.values

    return frequencies, power_spectrum


def dbfft(x, fs, win=None, ref=32768):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
        win: vector containing window samples (same length as x).
             If not provided, then rectangular window is used by default.
        ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """

    N = len(x)  # Length of input sequence

    if win is None:
        win = np.ones(N)
    if len(x) != len(win):
        raise ValueError('Signal and window must be of the same length')
    x = x * win

    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)

    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / np.sum(win)

    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag / ref)

    return freq, s_dbfs


def freq_IR(rpm, d, D, alpha, z):
    '''

    Funktion zur Berechnung der für charakteristische Frequenzen auf dem
    Innenring des Kugellagers.

    Parameters
    ----------
    rpm :   Drehgeschwindigkeit [1/min]
    d :     Rollkörperdurchmesser [mm]
    D :     Käfigdurchmesser    [mm]
    alpha : Druckwinkel        [°]

    Returns
    -------
    f_I : Frequenz für Innenringschäden [1/s]

    '''
    f_I = rpm * z / 2 * (1 + d * np.cos(alpha * 2 * np.pi / 360) / D)

    return f_I


def freq_OR(rpm, d, D, alpha, z):
    '''

    Funktion zur Berechnung der für charakteristische Frequenzen auf dem
    Außenring des Kugellagers.

    Parameters
    ----------
    rpm :   Drehgeschwindigkeit [1/min]
    d :     Rollkörperdurchmesser [mm]
    D :     Käfigdurchmesser    [mm]
    alpha : Druckwinkel

    Returns
    -------
    f_O : Frequenz für Innenringschäden [1/s]

    '''

    f_O = rpm * z / 2 * (1 - d * np.cos(alpha * 2 * np.pi / 360) / D)

    return f_O


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    '''


    Parameters
    ----------
    data : Data in time domain
    lowcut : f3dB for lower cut
    highcut : f3dB fpr higher cut
    fs : sampling frequency
    order : order of the bandpass

    Returns
    -------
    y : filtered signal

    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y

