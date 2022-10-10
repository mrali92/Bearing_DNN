from scipy.signal import find_peaks, peak_widths


def mean_feature(power, fs, fmin, fmax, teiler):
    freqTeil = int((fmax - fmin) / teiler)
    pTeil = []
    for i in range(0, teiler):
        anfang = int(len(power) / (fs / 2) * (fmin + i * freqTeil))
        avg = 0.0
        for j in range(0, freqTeil):
            avg = avg + power[anfang + j]
        pTeil.append(avg)
    return pTeil


def peaks_features(power, thershold, height, distance, rel_height, num_peaks):
    peak_idx, _ = find_peaks(power, height=height, threshold=thershold, distance=distance)
    peaks_widths = peak_widths(power, peak_idx, rel_height=rel_height)[0]  # save only the widths
    freqs = power[peak_idx]

    ranks = sorted([(x, i) for (i, x) in enumerate(peak_idx)], reverse=True)
    values = []
    posns = []

    for x, i in ranks:
        if x not in values:
            values.append(x)
            posns.append(i)
            if len(values) == num_peaks:
                break
    return freqs[posns], peak_idx[posns], peaks_widths[posns]
