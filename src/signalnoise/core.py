import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import blackmanharris


def load_npz_block(fname, block_index=5, remove_mean=True):
    """
    Load an .npz file that contains:
      - data : array shaped (nblocks, nsamples)
      - fs   : sample rate (Hz)

    Returns:
      x  (1D float array)
      fs (float)
    """
    loaded = np.load(fname)
    data = loaded["data"]
    fs = float(loaded["fs"])

    x = data[block_index].astype(float)
    if remove_mean:
        x = x - np.mean(x)

    return x, fs


def fit_sine_window(x, fs, f0, i0, i1):
    """
    Fit x[n] over indices [i0, i1) to:
      x[n] ≈ a*sin(w n) + b*cos(w n) + c
    Returns:
      a, b, c, n_fit (int array), x_fit (fit on integer indices)
    """
    i0 = int(i0)
    i1 = int(i1)

    n_fit = np.arange(i0, i1)
    x_win = x[i0:i1]

    w = 2*np.pi*f0/fs
    S = np.sin(w*n_fit)
    C = np.cos(w*n_fit)

    A_mat = np.column_stack([S, C, np.ones_like(n_fit)])
    a, b, c = np.linalg.lstsq(A_mat, x_win, rcond=None)[0]

    x_fit = a*S + b*C + c
    return a, b, c, n_fit, x_fit


def sine_from_coeffs(a, b, c, fs, f0, n):
    """
    Evaluate y(n) = a*sin(w n) + b*cos(w n) + c for any n (float or int).
    """
    w = 2*np.pi*f0/fs
    return a*np.sin(w*n) + b*np.cos(w*n) + c


def load_and_fit_sine(fname, f0,
                      block_index=5,
                      remove_mean=True,
                      xlim=(550, 580),
                      n_smooth=2000,
                      plot=True,
                      data_style=None,
                      fit_style=None,
                      title=None):
    """
    One-stop function:
      1) load x, fs from npz
      2) fit sine at f0 over the window given by xlim (in sample indices)
      3) return everything needed (and optionally plot)

    Inputs:
      - fname: npz file
      - f0: tone frequency (Hz)
      - block_index: which row of data[] to use
      - remove_mean: subtract mean from x
      - xlim: (i0, i1) sample-index window to fit/plot
      - n_smooth: number of points for smooth fit curve
      - plot: if True, makes the plot
      - data_style: dict of matplotlib kwargs for data points
      - fit_style: dict of matplotlib kwargs for fit curve

    Returns dict with:
      x, fs, f0, i0, i1, a, b, c, n_fit, x_fit, n_smooth, y_smooth
    """
    # ---------- load ----------
    x, fs = load_npz_block(fname, block_index=block_index, remove_mean=remove_mean)

    # ---------- window ----------
    i0, i1 = xlim
    a, b, c, n_fit, x_fit = fit_sine_window(x, fs, f0, i0, i1)

    # ---------- smooth curve ----------
    n_s = np.linspace(i0, i1, n_smooth)
    y_s = sine_from_coeffs(a, b, c, fs, f0, n_s)

    # ---------- plotting ----------
    if plot:
        if data_style is None:
            data_style = dict(marker='o', linestyle='none', color="royalblue",
                              markerfacecolor="royalblue", markeredgecolor="royalblue")
        if fit_style is None:
            fit_style = dict(color="limegreen", linewidth=1)

        plt.plot(x, label=f"{f0/1e3:.0f} kHz (data)", **data_style)
        plt.xlim(i0, i1)
        plt.ylim(-55, 55)  # change if needed

        plt.plot(n_s, y_s, label=f"{f0/1e3:.0f} kHz (local best-fit sine)", **fit_style)

        plt.legend()
        plt.xlabel("Counts")
        plt.ylabel("Voltage (arbitrary units)")
        #plt.title(title if title else "Wave data compared to fitted sine wave")
        plt.show()

    return {
        "x": x, "fs": fs, "f0": f0,
        "i0": int(i0), "i1": int(i1),
        "a": a, "b": b, "c": c,
        "n_fit": n_fit, "x_fit": x_fit,
        "n_smooth": n_s, "y_smooth": y_s
    }


def plot_clean_vs_noisy_psd(fname_clean, fname_noisy, f0,
                            block_index=5,
                            fmax_khz=None,
                            window="blackmanharris",
                            ax=None,
                            labels=("Clean (windowed)", "Noisy (windowed)"),
                            alpha_noisy=0.8):
    """
    Load clean+noisy npz files, compute windowed FFT power spectra, and plot them.
    - fname_clean, fname_noisy: filenames like "700khz(1).npz" and "700khzNoise.npz"
    - f0: expected tone frequency in Hz
    - block_index: which row of data[] to use
    - fmax_khz: x-axis max in kHz (defaults to Nyquist)
    - ax: pass a matplotlib axis to draw on; if None, creates a new figure.
    Returns: ax, (fpos_khz, P_clean_pos, P_noisy_pos)
    """

    # ---------- load clean ----------
    loaded = np.load(fname_clean)
    data_clean = loaded["data"]
    fs = loaded["fs"]
    x_clean = data_clean[block_index].astype(float)
    x_clean = x_clean - np.mean(x_clean)

    # ---------- load noisy ----------
    loaded = np.load(fname_noisy)
    data_noisy = loaded["data"]
    x_noisy = data_noisy[block_index].astype(float)
    x_noisy = x_noisy - np.mean(x_noisy)

    # ---------- window ----------
    N = len(x_clean)
    if window == "blackmanharris":
        w = blackmanharris(N, sym=False)
    else:
        w = np.ones(N)  # rectangular fallback

    # ---------- FFT + power ----------
    X_clean = np.fft.fft(x_clean * w)
    X_noisy = np.fft.fft(x_noisy * w)

    f = np.fft.fftfreq(N, d=1/fs)
    pos = f > 0

    fpos_khz = f[pos] / 1e3
    P_clean_pos = (np.abs(X_clean)**2)[pos]
    P_noisy_pos = (np.abs(X_noisy)**2)[pos]

    # ---------- plotting ----------
    if ax is None:
        fig, ax = plt.subplots()

    
    ax.semilogy(
        fpos_khz,
        P_clean_pos,
        color="royalblue",
        label=labels[0]
    )

    ax.semilogy(
        fpos_khz,
        P_noisy_pos,
        color="indianred",
        alpha=alpha_noisy,
        label=labels[1]
    )

    ax.axvline(f0/1e3, linestyle=":", color="k", label=f"Expected tone ({f0/1e3:.0f} kHz)")

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (arbitrary)")
    ax.grid()
    ax.legend()
    plt.show()

    if fmax_khz is None:
        fmax_khz = (fs/2)/1e3
    ax.set_xlim(0, fmax_khz)

    return ax, (fpos_khz, P_clean_pos, P_noisy_pos)


def plot_two_comparisons(pair1, pair2, same_axes=False, title=None, fmax_khz=None):
    """
    pair = (fname_clean, fname_noisy, f0)
    If same_axes=True, overlays both comparisons on one plot.
    If same_axes=False, makes two separate plots.
    """
    if same_axes:
        fig, ax = plt.subplots()
        plot_clean_vs_noisy_psd(*pair1, ax=ax, fmax_khz=fmax_khz,
                               labels=(f"{pair1[2]/1e3:.0f}k clean", f"{pair1[2]/1e3:.0f}k noisy"))
        plot_clean_vs_noisy_psd(*pair2, ax=ax, fmax_khz=fmax_khz,
                               labels=(f"{pair2[2]/1e3:.0f}k clean", f"{pair2[2]/1e3:.0f}k noisy"))
        ax.set_title(title if title else "Two clean vs noisy comparisons")
        ax.legend()
        plt.show()
    else:
        ax, _ = plot_clean_vs_noisy_psd(*pair1, fmax_khz=fmax_khz)
        ax.set_title(title if title else f"Clean vs Noisy PSD ({pair1[2]/1e3:.0f} kHz)")
        ax.legend()
        plt.show()

        ax, _ = plot_clean_vs_noisy_psd(*pair2, fmax_khz=fmax_khz)
        ax.set_title(title if title else f"Clean vs Noisy PSD ({pair2[2]/1e3:.0f} kHz)")
        ax.legend()
        plt.show()


def plot_power_spectrum(fname, block_index=5, fmax_khz=1500, remove_mean=True, use_logy=False):
    """
    Load an .npz file (expects keys: 'data' and 'fs'), compute FFT power spectrum,
    and plot power vs frequency for positive frequencies only.

    Inputs:
      fname        : e.g. "700khz(1).npz"
      block_index  : which row of data[] to use (default 5)
      fmax_khz     : x-axis max in kHz (default 1500)
      remove_mean  : subtract DC offset before FFT (default True)
      use_logy     : if True, semilogy plot (default False)

    Returns:
      f_pos (Hz), P_pos (power)
    """
    loaded = np.load(fname)
    data = loaded["data"]
    fs   = float(loaded["fs"])

    x = data[block_index].astype(float)
    if remove_mean:
        x = x - np.mean(x)

    N = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(N, d=1/fs)
    P = np.abs(X)**2

    mask = f > 0
    f_pos = f[mask]
    P_pos = P[mask]

    if use_logy:
        plt.semilogy(f_pos/1e3, P_pos, color = "royalblue")
    else:
        plt.plot(f_pos/1e3, P_pos, color = "royalblue")

    plt.xlabel("Frequency (kHz)")
    plt.ylabel("log10 Power (Arbitrary Units)")
    #plt.title(f"Power Spectrum ({fname})")
    plt.xlim(0, fmax_khz)
    plt.show()

    return f_pos, P_pos

def plot_voltage_spectrum(fname,
                          block_index=5,
                          remove_mean=True,
                          use_logy=True,
                          title_suffix=""):
    """
    Plot the voltage (magnitude) spectrum |FFT(x)| vs frequency-bin index.

    Inputs:
      fname        : e.g. "700khz(1).npz"
      block_index  : which row of data[] to use
      remove_mean  : subtract DC offset before FFT
      use_logy     : semilogy if True, linear if False
      color        : line color
      title_suffix : optional string to add to title

    Returns:
      |X| (FFT magnitude)
    """
    loaded = np.load(fname)
    data = loaded["data"]
    fs   = loaded["fs"]

    x = data[block_index].astype(float)
    if remove_mean:
        x = x - np.mean(x)

    X = np.fft.fft(x)
    mag = np.abs(X)

    if use_logy:
        plt.semilogy(mag, color="royalblue")
        plt.ylabel("log10 Voltage (arbitrary units)")
    else:
        plt.plot(mag, color="royalblue")
        plt.ylabel("Voltage (arbitrary units)")

    plt.xlabel("FFT bin (counts)")
    #plt.title(f"Voltage Spectrum {title_suffix}")
    plt.grid()
    plt.show()

    return mag

from scipy import stats

def chi2_gaussianity_from_file(fname, f0,
                              block_index=5,
                              remove_mean=True,
                              fit_window=None,
                              bins=100,
                              xmin=-100, xmax=100,
                              min_expected=5,
                              plot=True,
                              label=None):
    """
    End-to-end chi-square GOF test: residuals vs Gaussian.
    Residuals are computed as: x - best_fit_sine_at_f0

    Returns dict with:
      mu, sigma, chi2_stat, chi2_p, dof, bins_used, N_residual
    """

    # Load data 
    loaded = np.load(fname)
    data = loaded["data"]
    fs = float(loaded["fs"])

    x = data[block_index].astype(float)
    if remove_mean:
        x = x - np.mean(x)

    N = len(x)
    n = np.arange(N)

    # Choose fit window 
    if fit_window is None:
        i0, i1 = 0, N
    else:
        i0, i1 = int(fit_window[0]), int(fit_window[1])

    n_fit = n[i0:i1]
    x_fitdata = x[i0:i1]

    # Fit sine at f0: x ≈ a*sin(w n) + b*cos(w n) + c 
    w = 2*np.pi*f0/fs
    S = np.sin(w*n_fit)
    C = np.cos(w*n_fit)
    A_mat = np.column_stack([S, C, np.ones_like(n_fit)])
    a, b, c = np.linalg.lstsq(A_mat, x_fitdata, rcond=None)[0]

    # Model across full record, residual across full record
    x_model = a*np.sin(w*n) + b*np.cos(w*n) + c
    residual = x - x_model
    residual = residual[np.isfinite(residual)]
    Nr = len(residual)

    # Fit Gaussian params from residual 
    mu = np.mean(residual)
    sigma = np.std(residual, ddof=1)

    # Chi-square GOF on histogram 
    counts, edges = np.histogram(residual, bins=bins, range=(xmin, xmax))

    # Expected probability mass per bin under Normal(mu, sigma)
    cdf_hi = stats.norm.cdf(edges[1:], loc=mu, scale=sigma)
    cdf_lo = stats.norm.cdf(edges[:-1], loc=mu, scale=sigma)
    p_bin = cdf_hi - cdf_lo
    expected = Nr * p_bin

    # Keep only bins with enough expected counts
    good = expected >= min_expected
    obs_good = counts[good]
    exp_good = expected[good]

    # dof = (#bins_used - 1 - #fit_params). We fit mu & sigma => 2 params
    bins_used = int(np.sum(good))
    dof = bins_used - 1 - 2

    if dof > 0 and np.all(exp_good > 0):
        chi2_stat = np.sum((obs_good - exp_good)**2 / exp_good)
        chi2_p = stats.chi2.sf(chi2_stat, dof)
    else:
        chi2_stat, chi2_p = np.nan, np.nan

    # Optional plot: histogram + Gaussian overlay
    if plot:
        plot_label = label if label is not None else fname

        plt.hist(residual, bins=bins, range=(xmin, xmax), alpha=0.7,
                 label=f"{plot_label} residuals", color = "royalblue", edgecolor="blue", lw=0.5)

        xx = np.linspace(xmin, xmax, 2000)
        pdf = stats.norm.pdf(xx, loc=mu, scale=sigma)

        # scale pdf to expected histogram counts
        bin_width = edges[1] - edges[0]
        gauss_counts = pdf * Nr * bin_width

        plt.plot(xx, gauss_counts, linewidth=2,
                 label=f"Gaussian fit: μ={mu:.2f}, σ={sigma:.2f}", color = "limegreen")

        plt.xlabel("Residual value (arb units)")
        plt.ylabel("Counts")
        #plt.title(f"Residual histogram + chi-square GOF (p={chi2_p:.3g})")
        plt.legend(fontsize=8)
        plt.show()

    return {
        "file": fname,
        "fs": fs,
        "f0_hz": f0,
        "block_index": block_index,
        "fit_window": (i0, i1),
        "N_residual": Nr,
        "mu": mu,
        "sigma": sigma,
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "chi2_dof": dof,
        "chi2_bins_used": bins_used
    }



def estimate_aliased_tone_clean(fname_clean,
                                block_index=5,
                                remove_mean=True,
                                window="blackmanharris",
                                search_width_hz=None,
                                expected_alias_hz=None,
                                true_band_hz=None,
                                max_fold=20,
                                plot=True,
                                fmax_khz=None):
    """
    Estimate the observed (aliased) tone frequency from CLEAN data only,
    then return possible original analog frequencies that could produce that alias.

    Inputs
    ------
    fname_clean            : .npz file with keys ["data", "fs"]
    block_index            : which row in data[] to use
    remove_mean            : subtract DC offset
    window                 : "blackmanharris" or "rect"
    expected_alias_hz      : expected aliased frequency (Hz), if known
    search_width_hz        : half-width for peak search around expected_alias_hz (Hz)
    true_band_hz           : (fmin, fmax) plausible TRUE analog frequency band (Hz)
    max_fold               : number of fs-folds to consider if true_band_hz not given
    plot                   : plot PSD + detected alias
    fmax_khz               : x-axis limit in kHz (defaults to Nyquist)

    Returns
    -------
    dict with:
      fs,
      f_alias_hz,
      candidates_hz
    """

    # Load clean data
    loaded = np.load(fname_clean)
    data = loaded["data"]
    fs = float(loaded["fs"])

    x = data[block_index].astype(float)
    if remove_mean:
        x = x - np.mean(x)

    # Window + FFT
    N = len(x)
    if window == "blackmanharris":
        w = blackmanharris(N, sym=False)
    else:
        w = np.ones(N)

    X = np.fft.fft(x * w)
    f = np.fft.fftfreq(N, d=1/fs)

    pos = f > 0
    fpos = f[pos]
    P = (np.abs(X)**2)[pos]

    # Find aliased peak
    if expected_alias_hz is not None and search_width_hz is not None:
        mask = np.abs(fpos - expected_alias_hz) <= search_width_hz
        if not np.any(mask):
            raise ValueError("Search mask empty, widen search_width_hz.")
        idx = np.argmax(P[mask])
        f_alias = fpos[mask][idx]
    else:
        idx = np.argmax(P)
        f_alias = fpos[idx]

    f_alias = float(f_alias)

    # Build candidate true frequencies
    candidates = []

    if true_band_hz is not None:
        fmin, fmax = true_band_hz
        k_min = int(np.floor(fmin / fs))
        k_max = int(np.ceil(fmax / fs))

        for k in range(max(0, k_min-1), k_max+1):
            c1 = k*fs + f_alias
            c2 = (k+1)*fs - f_alias
            for c in (c1, c2):
                if fmin <= c <= fmax:
                    candidates.append(float(c))

        candidates = sorted(set(np.round(candidates, 12)))

    else:
        for k in range(max_fold+1):
            candidates.append(k*fs + f_alias)
            if k > 0:
                candidates.append(k*fs - f_alias)

        candidates = sorted(set(c for c in candidates if c >= 0))

    # Plot (clean only)
    if plot:
        if fmax_khz is None:
            fmax_khz = (fs/2)/1e3

        plt.semilogy(fpos/1e3, P, color="royalblue", label="Clean (windowed)")
        plt.axvline(f_alias/1e3, color="k", linestyle=":",
                    label=f"Observed alias ≈ {f_alias/1e3:.1f} kHz")
        plt.xlabel("Frequency (kHz)")
        plt.ylabel("Power (arb units)")
        #plt.title("Aliased tone identification")
        plt.xlim(0, fmax_khz)
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "fs": fs,
        "f_alias_hz": f_alias,
        "candidates_hz": candidates
    }


import ugradio
fs = 3e6
nsamples = 2048
nblocks = 10
sdr = ugradio.sdr.SDR(sample_rate=fs, fir_coeffs=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2047]))
data = sdr.capture_data(nsamples, nblocks=nblocks)
print("data shape:", data.shape)
print("dtype:", data.dtype)
print("fs (Hz):", fs)
np.savez("1500khzNoise.npz", data=data, fs=fs)
