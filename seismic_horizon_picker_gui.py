# =============================================================================
# Seismic Horizon Picker GUI - Version 1
# =============================================================================
# Python desktop application for seismic (SBP/SEG-Y) data interpretation
# =============================================================================
# Version 1
# E.Tripoliti 12/06/2025 ucfbetr@ucl.ac.uk
# =============================================================================

# ---------------------------- IMPORTS AND SETUP ------------------------------
import sys
import os
import segyio
import numpy as np
from scipy.signal import hilbert, butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter1d
import matplotlib
matplotlib.use('Qt5Agg')  # Use for compatibility
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QMessageBox, QProgressBar,
    QInputDialog, QDialog, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.cluster import DBSCAN
from collections import defaultdict
import multiprocessing
import pandas as pd
from scipy.signal import find_peaks

# -------------------------- UTILITY FUNCTIONS -------------------------------
def envelope_to_grayscale(envelope):
    """
    Convert a seismic envelope matrix to an 8-bit grayscale image using
    a robust scaling based on mean ± 3*std (Haralick transform).
    """
    mu = np.mean(envelope)
    delta = np.std(envelope)
    Boundup = mu + 3 * delta
    Boundlow = mu - 3 * delta
    G = (envelope - Boundlow) / (Boundup - Boundlow) * 255
    G = np.clip(G, 0, 255)
    return G.astype(np.uint8)

def twt_to_depth(twt_array, velocity=1500.0):
    """
    Convert two-way travel-time (ms) to depth (m) using acoustic velocity.
    Default velocity is 1500 m/s (typical seawater).
    """
    return (velocity * twt_array / 1000.0) / 2.0

def depth_to_twt(depths, velocity=1500.0): # velocity can change, suggested for marine sediments from 1.4 km/s - 1.8 km/s
    """
    Convert depth (m) to two-way travel-time (ms) using acoustic velocity.
    """
    return (np.array(depths) * 2 * 1000.0) / velocity

def butter_filter(data, fs, kind='low', cutoff=None, order=4, band=None):
    """
    Apply a Butterworth filter (low, high, or band) to a 1D signal.
    - data: 1D input array.
    - fs: Sampling frequency (Hz).
    - kind: 'low', 'high', or 'band'.
    - cutoff/band: cutoff frequency/frequencies in Hz.
    """
    nyq = 0.5 * fs
    if kind == 'low':
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low')
    elif kind == 'high':
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
    elif kind == 'band':
        normal_band = [f / nyq for f in band]
        b, a = butter(order, normal_band, btype='band')
    else:
        return data
    y = filtfilt(b, a, data)
    return y

def robust_orientation(trace_indices, sample_indices, window_size=11, min_points=5, interp_gaps=True):
    """
    Compute local dip and azimuth along the main horizon using robust linear regression
    within a moving window. Handles gaps by interpolation, and smooths outliers.
    """
    tr = np.array(trace_indices)
    sm = np.array(sample_indices)
    sort_idx = np.argsort(tr)
    tr = tr[sort_idx]
    sm = sm[sort_idx]
    # Interpolate gaps for continuity
    if interp_gaps and (np.diff(tr).max() > 1):
        all_tr = np.arange(tr[0], tr[-1]+1)
        all_sm = np.interp(all_tr, tr, sm)
        tr, sm = all_tr, all_sm
    # Median filter for outlier suppression
    win = min(window_size, len(sm)//2*2+1)
    sm = median_filter(sm, size=win if win > 2 else 1)
    dip = np.full_like(tr, np.nan, dtype=float)
    azimuth = np.full_like(tr, np.nan, dtype=float)
    half_w = window_size // 2
    # Moving window regression for dip/azimuth
    for i in range(len(tr)):
        lo = max(0, i - half_w)
        hi = min(len(tr), i + half_w + 1)
        t_win = tr[lo:hi]
        s_win = sm[lo:hi]
        if len(t_win) < min_points:
            continue
        A = np.vstack([t_win, np.ones_like(t_win)]).T
        m, c = np.linalg.lstsq(A, s_win, rcond=None)[0]
        dip[i] = np.arctan(m)
        azimuth[i] = (np.degrees(np.arctan2(m, 1)) + 360) % 360
    return tr, sm, dip, azimuth

def local_adaptive_maxima(env, window=51, min_dist=10):
    peaks = []
    n = len(env)
    half_win = window // 2
    for i in range(half_win, n-half_win):
        win = env[i-half_win:i+half_win+1]
        local_thresh = np.percentile(win, 90)
        if env[i] == np.max(win) and env[i] > local_thresh:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
    return np.array(peaks)

def pick_trace(env, cos_phase, min_sample, max_sample, max_horizons, min_dist, thresh_ratio=0.9):
    env = np.asarray(env)
    env_win = env[min_sample:max_sample+1]
    if len(env_win) < 5:
        return []
    # Use the user-supplied percentile as a threshold
    threshold = np.percentile(env_win, 100 * thresh_ratio)
    peaks, _ = find_peaks(env_win, height=threshold, distance=min_dist)
    peaks = peaks + min_sample
    zero_crossings = np.where(np.diff(np.sign(cos_phase)))[0]
    picks = []
    for m in peaks[:max_horizons]:
        if len(zero_crossings) == 0:
            continue
        closest = min(zero_crossings, key=lambda z: abs(z - m))
        if abs(closest - m) <= 3:
            picks.append(closest)
    return picks

def extract_all_segments(pick_matrix, min_length=20):
    """
    Given a 1D pick array (shape: num_traces,), find all continuous segments
    of valid picks (not None) that are at least min_length long.
    Returns: list of lists of (trace_idx, sample_idx)
    """
    segments = []
    current = []
    for i, val in enumerate(pick_matrix):
        if val is not None and val >= 0:
            current.append((i, int(val)))
        else:
            if len(current) >= min_length:
                segments.append(current)
            current = []
    if len(current) >= min_length:
        segments.append(current)
    return segments
def track_horizon_advanced(
    envelope,
    cos_phase,
    seed_trace,
    seed_sample,
    forbidden_indices=None,
    window=10,  # Number of samples above/below previous pick to search for the next pick (vertical search window)
    max_dip=8,  # Maximum allowed jump (in samples) between consecutive picks (limits horizon steepness)
    min_snr=2.0,  # Minimum signal-to-noise ratio required for a valid pick (peak must be at least 2x local noise)
    min_length=10,  # Minimum number of continuous picks (traces) for a segment to be accepted as a valid horizon
):
    """
    Overlap-safe horizon tracker: forbidden_indices is checked for every possible pick.
    """
    num_traces, num_samples = envelope.shape
    picks = [None] * num_traces
    picks[seed_trace] = seed_sample
    prev_sample = seed_sample

    # Forward
    for t in range(seed_trace + 1, num_traces):
        search_min = max(0, prev_sample - window)
        search_max = min(num_samples, prev_sample + window + 1)
        segment_env = envelope[t, search_min:search_max]
        segment_cos = cos_phase[t, search_min:search_max]
        if segment_env.size == 0:
            picks[t] = None
            continue
        rel_max = int(np.argmax(segment_env))
        pick_candidate = search_min + rel_max
        # NO OVERLAP: do not pick if forbidden
        if forbidden_indices and pick_candidate in forbidden_indices[t]:
            picks[t] = None
            continue
        local_noise = np.median(segment_env)
        max_amp = segment_env[rel_max]
        if max_amp < min_snr * (local_noise + 1e-6):
            picks[t] = None
            continue
        zeros = np.where(np.diff(np.sign(segment_cos)))[0]
        if len(zeros) == 0:
            pick = search_min + rel_max
        else:
            pick = search_min + min(zeros, key=lambda z: abs(z - rel_max))
        if abs(pick - prev_sample) > max_dip:
            picks[t] = None
            continue
        # NO OVERLAP: do not pick if forbidden (after zero-cross)
        if forbidden_indices and pick in forbidden_indices[t]:
            picks[t] = None
            continue
        picks[t] = pick
        prev_sample = pick

    # Backward
    prev_sample = seed_sample
    for t in range(seed_trace - 1, -1, -1):
        search_min = max(0, prev_sample - window)
        search_max = min(num_samples, prev_sample + window + 1)
        segment_env = envelope[t, search_min:search_max]
        segment_cos = cos_phase[t, search_min:search_max]
        if segment_env.size == 0:
            picks[t] = None
            continue
        rel_max = int(np.argmax(segment_env))
        pick_candidate = search_min + rel_max
        if forbidden_indices and pick_candidate in forbidden_indices[t]:
            picks[t] = None
            continue
        local_noise = np.median(segment_env)
        max_amp = segment_env[rel_max]
        if max_amp < min_snr * (local_noise + 1e-6):
            picks[t] = None
            continue
        zeros = np.where(np.diff(np.sign(segment_cos)))[0]
        if len(zeros) == 0:
            pick = search_min + rel_max
        else:
            pick = search_min + min(zeros, key=lambda z: abs(z - rel_max))
        if abs(pick - prev_sample) > max_dip:
            picks[t] = None
            continue
        if forbidden_indices and pick in forbidden_indices[t]:
            picks[t] = None
            continue
        picks[t] = pick
        prev_sample = pick

    # Minimum length constraint
    indices = [i for i, p in enumerate(picks) if p is not None]
    if not indices:
        return []
    segments = []
    current = []
    for idx in indices:
        if not current or idx == current[-1] + 1:
            current.append(idx)
        else:
            segments.append(current)
            current = [idx]
    if current:
        segments.append(current)
    longest = max(segments, key=len)
    if len(longest) < min_length:
        return []
    return [(t, int(picks[t])) for t in longest]

# ------------------- WORKER THREAD FOR HORIZON PICKING ----------------------
class HorizonWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, seismic_data, sample_depths, params):
        super().__init__()
        self.seismic_data = seismic_data
        self.sample_depths = sample_depths
        self.params = params

    def run(self):
        try:
            self.progress.emit(5)
            analytic_signal = hilbert(self.seismic_data, axis=1)
            envelope = np.abs(analytic_signal)
            phase = np.angle(analytic_signal)
            self.progress.emit(15)
            envelope_smoothed = gaussian_filter(envelope, sigma=self.params['sigma'])
            self.progress.emit(25)
            num_traces, num_samples = envelope_smoothed.shape
            cos_phase = np.cos(phase)
            n_horizons = self.params['max_horizons']

            min_horizon_spacing = 20
            trace_mask_width = 2
            min_continuity = 5

            env_work = envelope_smoothed.copy()
            forbidden_indices = [set() for _ in range(num_traces)]
            horizons = []

            for horizon_idx in range(n_horizons):
                for t in range(num_traces):
                    for s_forbid in forbidden_indices[t]:
                        env_work[t, s_forbid] = 0

                # --------- STRICT ORDERING BLOCK -----------
                # Find the deepest pick of all previous horizons at each trace
                min_allowed = [0] * num_traces
                for t in range(num_traces):
                    # Find the maximum pick among all previous horizons at this trace
                    prev_picks = [h[t] for h in horizons if t < len(h) and h[t] is not None]
                    if prev_picks:
                        min_allowed[t] = max(prev_picks) + 1  # strictly below all previous
                    else:
                        min_allowed[t] = 0
                    env_work[t, :min_allowed[t]] = 0
                # -------------------------------------------

                if np.all(env_work == 0):
                    break

                seed_trace, seed_sample = np.unravel_index(np.argmax(env_work), env_work.shape)
                if env_work[seed_trace, seed_sample] == 0:
                    break

                # -----Tracking Parameters-----
                raw_picks = track_horizon_advanced(
                    env_work,
                    cos_phase,
                    seed_trace,
                    seed_sample,
                    forbidden_indices=forbidden_indices,
                    window=15,
                    max_dip=15,
                    min_snr=1.0,
                    min_length=5
                )

                picks = []
                for t, s in raw_picks:
                    if (
                            s is not None and
                            0 <= t < num_traces and
                            0 <= s < num_samples and
                            env_work[t, s] > 0 and
                            s not in forbidden_indices[t]
                    ):
                        # Enforce again: only allow strictly below all previous
                        prev_picks = [h[t] for h in horizons if t < len(h) and h[t] is not None]
                        if prev_picks and s <= max(prev_picks):
                            continue
                        picks.append((t, s))

                if len(picks) < min_continuity:
                    break

                pick_samples = [s for t, s in picks]
                if len(pick_samples) == 0 or np.std(pick_samples) < 2:
                    continue
                if (max(pick_samples) - min(pick_samples)) < 3:
                    continue
                env_along_horizon = [env_work[t, s] for t, s in picks]
                if np.mean(env_along_horizon) < np.percentile(envelope_smoothed, 50):
                    continue

                horizon_picks = [None] * num_traces
                for t, s in picks:
                    horizon_picks[t] = s
                    if s is not None:
                        forbidden_indices[t].add(s)

                horizons.append(horizon_picks)

                # Mask above/below and laterally for next horizon
                for t, s in picks:
                    if s is not None:
                        for tt in range(max(0, t - trace_mask_width), min(num_traces, t + trace_mask_width + 1)):
                            lo = max(0, s - min_horizon_spacing)
                            hi = min(num_samples, s + min_horizon_spacing + 1)
                            env_work[tt, lo:hi] = 0

                self.progress.emit(int(25 + 25 * (horizon_idx + 1) / n_horizons))

            # Deduplicate: only first horizon per trace/sample
            used = [set() for _ in range(num_traces)]
            for h in horizons:
                for t, s in enumerate(h):
                    if s is not None:
                        if s in used[t]:
                            h[t] = None
                        else:
                            used[t].add(s)

            main_horizons = horizons

            orientations = []
            for horizon in main_horizons:
                trace_indices = [i for i, s in enumerate(horizon) if s is not None]
                sample_indices = [s for s in horizon if s is not None]
                if len(trace_indices) < 3:
                    orientations.append(None)
                    continue
                tr, sm, dip, azimuth = robust_orientation(
                    trace_indices, sample_indices,
                    window_size=self.params.get('orientation_window', 7),
                    min_points=3, interp_gaps=True)
                orientations.append({'trace_indices': tr, 'sample_indices': sm, 'dip': dip, 'azimuth': azimuth})

            self.progress.emit(100)
            self.finished.emit({
                'horizons': main_horizons,
                'orientations': orientations,
                'envelope': envelope_smoothed,
                'phase': phase
            })
        except Exception as e:
            self.error.emit(str(e))

class CosinePhaseHorizonsDialog(QDialog):
    def __init__(self, phase, auto_picks, manual_picks_dict=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cosine Phase + Horizons")
        layout = QVBoxLayout(self)
        fig, ax = plt.subplots(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        cos_phase = np.cos(phase)  # shape: [num_traces, num_samples]
        ax.imshow(
            cos_phase.T,  # traces on x, samples on y
            cmap="gray",
            aspect="auto",
            origin="upper"
        )
        # Overlay auto picks (all traces)
        if auto_picks:
            pick_i, pick_j = zip(*auto_picks)
            ax.plot(pick_i, pick_j, 'rx', label="Auto Pick", markersize=3, alpha=0.9)
        # Overlay manual picks (all traces) -- CYAN CIRCLES
        if manual_picks_dict:
            manual_trace_idxs = []
            manual_sample_idxs = []
            for trc_idx, picks in manual_picks_dict.items():
                for sample_idx in picks:
                    manual_trace_idxs.append(trc_idx)
                    manual_sample_idxs.append(sample_idx)
            if manual_trace_idxs:
                ax.plot(manual_trace_idxs, manual_sample_idxs, 'o', color='c', markersize=5, alpha=0.8, label="Manual Pick")
        ax.set_xlabel("Trace Index")
        ax.set_ylabel("Sample Index (TWT)")
        ax.set_title("Cosine Phase Image with All Horizons")
        ax.legend()
        fig.tight_layout()
        canvas.draw()
# ----------------- DIP/AZIMUTH, REFLECTION COEFFICIENT, WAVEFORM ----------------
class DipAzimuthDialog(QDialog):
    """
    Visualizing dip magnitude, azimuth, and rose diagram
    for the main picked seismic horizon.
    """
    def __init__(self, grouped_horizons, local_orientations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dip, Azimuth, and Rose Diagram")
        layout = QVBoxLayout(self)
        # Dip/Azimuth main plot
        self.fig = Figure(figsize=(12, 9), constrained_layout=True)
        self.ax_dip = self.fig.add_subplot(211)
        self.ax_az = self.fig.add_subplot(212)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # Rose diagram
        self.fig2 = Figure(figsize=(6, 6), constrained_layout=True)
        self.rose_ax = self.fig2.add_subplot(111, polar=True)
        self.canvas2 = FigureCanvas(self.fig2)
        layout.addWidget(self.canvas2)
        # Compute and plot dip, azimuth, rose
        try:
            has_data = bool(grouped_horizons) and bool(local_orientations)
            if has_data:
                # Select main horizon (the one with the most non-None picks)
                main_idx = np.argmax([sum(s is not None for s in g) for g in grouped_horizons])
                main_horizon = grouped_horizons[main_idx]
                trace_indices = np.array([i for i, s in enumerate(main_horizon) if s is not None])
                sample_indices = np.array([s for s in main_horizon if s is not None])
                valid = (sample_indices >= 0) & (sample_indices < 1e6) & (~np.isnan(sample_indices))
                trace_indices = trace_indices[valid]
                sample_indices = sample_indices[valid]
                if len(trace_indices) > 2:
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(trace_indices, sample_indices, kind='linear', fill_value="extrapolate")
                    full_traces = np.arange(trace_indices[0], trace_indices[-1]+1)
                    smooth_samples = interp_func(full_traces)
                    smooth_samples = median_filter(smooth_samples, size=15)
                    tr, sm, dip, azimuth = robust_orientation(full_traces, smooth_samples, window_size=21)
                    dip_vals = dip
                    azimuth_vals = azimuth
                    trace_indices_plot = tr
                    # Circular smoothing for azimuth (avoid 0/360 jumps)
                    azimuth_rad = np.deg2rad(azimuth_vals)
                    az_sin = np.sin(azimuth_rad)
                    az_cos = np.cos(azimuth_rad)
                    smooth_az_sin = uniform_filter1d(az_sin, size=21, mode='nearest')
                    smooth_az_cos = uniform_filter1d(az_cos, size=21, mode='nearest')
                    smooth_azimuth_rad = np.arctan2(smooth_az_sin, smooth_az_cos)
                    smooth_azimuth_deg = np.rad2deg(smooth_azimuth_rad) % 360
                else:
                    dip_vals = np.array([])
                    azimuth_vals = np.array([])
                    trace_indices_plot = np.array([])
                    smooth_azimuth_deg = np.array([])
                if len(dip_vals) > 0:
                    # Plot dip magnitude
                    self.ax_dip.plot(trace_indices_plot, dip_vals, color='k', lw=0.75)
                    self.ax_dip.set_title("Dip Magnitude Along Main Horizon", fontsize=14)
                    self.ax_dip.set_ylabel("Dip Magnitude (radians)", fontsize=12)
                    self.ax_dip.tick_params(axis='both', labelsize=10)
                    # Plot azimuth
                    self.ax_az.plot(trace_indices_plot, smooth_azimuth_deg, color='r', lw=0.75, label="Smoothed Azimuth")
                    self.ax_az.set_title("Azimuth Along Main Horizon (Smoothed)", fontsize=14)
                    self.ax_az.set_xlabel("Trace Index", fontsize=12)
                    self.ax_az.set_ylabel("Azimuth (degrees)", fontsize=12)
                    self.ax_az.tick_params(axis='both', labelsize=10)
                    self.ax_az.legend(fontsize=10)
                    # Rose diagram (histogram of azimuth)
                    azimuth_rad_smoothed = np.deg2rad(smooth_azimuth_deg)
                    self.rose_ax.hist(azimuth_rad_smoothed, bins=36, color='r', alpha=0.7)
                    self.rose_ax.set_title('Azimuth Rose Diagram', va='bottom', fontsize=13)
                else:
                    self.ax_dip.set_title("Dip Magnitude Along Main Horizon (No Data)", fontsize=14)
                    self.ax_az.set_title("Azimuth Along Main Horizon (No Data)", fontsize=14)
                    self.rose_ax.set_title('Azimuth Rose Diagram (No Data)', fontsize=13)
            else:
                self.ax_dip.set_title("Dip Magnitude Along Main Horizon (No Data)", fontsize=14)
                self.ax_az.set_title("Azimuth Along Main Horizon (No Data)", fontsize=14)
                self.rose_ax.set_title('Azimuth Rose Diagram (No Data)', fontsize=13)
            self.canvas.draw()
            self.canvas2.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot dip/azimuth: {e}")
        # Enable 'Save' buttons for exporting figures
        save_layout = QHBoxLayout()
        self.save_dip_az_btn = QPushButton("Save Dip/Azimuth (PNG, 900 DPI)")
        self.save_rose_btn = QPushButton("Save Rose Diagram (PNG, 900 DPI)")
        self.save_dip_az_btn.clicked.connect(self.save_dip_az_fig)
        self.save_rose_btn.clicked.connect(self.save_rose_fig)
        save_layout.addWidget(self.save_dip_az_btn)
        save_layout.addWidget(self.save_rose_btn)
        layout.addLayout(save_layout)
    def save_dip_az_fig(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Dip/Azimuth Figure", "", "PNG Image (*.png)")
        if fname:
            self.fig.savefig(fname, dpi=900, bbox_inches='tight')
    def save_rose_fig(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Rose Diagram", "", "PNG Image (*.png)")
        if fname:
            self.fig2.savefig(fname, dpi=900, bbox_inches='tight')

class ReflectionCoeffDialog(QDialog):
    """
    Displaying and exporting reflection coefficients
    for manual picks on a selected seismic trace.
    """
    def __init__(self, trace, sample_depths, picks, velocity=1500.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reflection Coefficient Analysis")
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        self.trace = trace
        self.sample_depths = sample_depths
        self.picks = picks
        self.velocity = velocity

        # Convert TWT (ms) to depth (m) for pick locations
        depths = twt_to_depth(sample_depths, velocity)
        pick_depths = [depths[idx] for idx in picks]

        # Reference pick selection for normalization
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference Pick:"))
        self.ref_combo = QComboBox()
        for i, d in enumerate(pick_depths):
            self.ref_combo.addItem(f"{i}: {d:.2f} m")
        ref_layout.addWidget(self.ref_combo)
        layout.addLayout(ref_layout)

        # Main plot setup
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)  # Toolbar goes here
        layout.addWidget(self.canvas)  # Then the canvas

        self.ref_combo.currentIndexChanged.connect(self.replot)
        self.replot()

    def compute_reflection_coefficients(self, trace, picks, window_samples=20, reference_idx=0):
        """
        Compute normalized reflection coefficients for each pick, relative to
        the amplitude at the reference pick.
        """
        coeffs = []
        ref_pick = picks[reference_idx]
        ref_amp = np.max(np.abs(trace[max(0, ref_pick-window_samples//2):ref_pick+window_samples//2+1]))
        for idx in picks:
            win_lo = max(0, idx - window_samples//2)
            win_hi = min(len(trace), idx + window_samples//2 + 1)
            amp = np.max(np.abs(trace[win_lo:win_hi]))
            coeffs.append(amp / ref_amp if ref_amp != 0 else 0)
        return coeffs

    def replot(self):
        """
        Draw (or re-draw) the reflection coefficient plot.
        """
        window_samples = 20
        reference_idx = self.ref_combo.currentIndex()
        coeffs = self.compute_reflection_coefficients(self.trace, self.picks, window_samples, reference_idx)
        depths = twt_to_depth(self.sample_depths, self.velocity)
        pick_depths = [depths[idx] for idx in self.picks]
        self.ax.clear()
        self.ax.plot(coeffs, pick_depths, 'o-', label="Reflection Coefficient")
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Normalized Reflection Coefficient")
        self.ax.set_ylabel("Depth (m)")
        self.ax.set_title("Reflection Coefficient vs Depth")
        self.ax.grid()
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

class TracePicksDialog(QDialog):
    def __init__(self, trace_idx, phase, manual_picks, auto_picks, sample_depths, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Manual & Auto Picks on Cosine Phase (Trace {trace_idx})")
        layout = QVBoxLayout(self)
        fig, ax = plt.subplots(figsize=(5, 8))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        # Plot cosine phase for the trace (vertical column)
        cos_phase = np.cos(phase[trace_idx])
        ax.imshow(
            cos_phase[:, np.newaxis].T,
            cmap="gray",
            aspect="auto",
            extent=[0, 1, 0, len(cos_phase)]
        )
        # Overlay manual picks
        for idx in manual_picks:
            ax.plot(0.5, idx, marker='x', color='c', markersize=10, label="Manual Pick" if idx == manual_picks[0] else "")
        # Overlay automatic picks
        for idx in auto_picks:
            ax.plot(0.5, idx, marker='x', color='r', markersize=8, label="Auto Pick" if idx == auto_picks[0] else "")
        ax.set_ylim(len(cos_phase), 0)
        ax.set_xlim(0, 1)
        ax.set_ylabel("Sample Index")
        ax.set_xticks([])
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        fig.tight_layout()
        canvas.draw()

class WaveformDialog(QDialog):
    """
    Visualizing and manually picking peaks on a single seismic trace.
    Provides filtering tools, manual pick controls, and interactive plot export.
    """
    def __init__(self, trace, sample_depths, trace_idx, parent=None, manual_picks_dict=None, max_sample_index=None):
        super().__init__(parent)
        self.setWindowTitle(f"Waveform at Trace {trace_idx}")
        self.trace = trace
        self.sample_depths = sample_depths
        self.trace_idx = trace_idx
        self.filtered_trace = trace.copy()
        self.fs = self.estimate_sampling_rate()
        self.manual_picks = []
        self.max_sample_index = max_sample_index
        self.xlim = None  # for zooming
        if manual_picks_dict is None:
            self.manual_picks_dict = {}
        else:
            self.manual_picks_dict = manual_picks_dict
        self.initUI()

    def estimate_sampling_rate(self):
        """
        Estimate sampling rate (Hz) from sample depth array.
        """
        diffs = np.diff(self.sample_depths)
        median_dt = np.median(diffs)
        if median_dt <= 0:
            return 1000.0
        return 1000.0 / median_dt

    def initUI(self):
        """
        Build the waveform dialog GUI: filter controls, plot, toolbar, and export buttons.
        """
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Low-pass", "High-pass", "Band-pass", "Smoothing"])
        controls.addWidget(self.filter_combo)
        # Filter parameter controls
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.01, max(0.01, self.fs / 2000 - 0.01))
        self.cutoff_spin.setSingleStep(0.1)
        self.cutoff_spin.setValue(1.0)
        self.cutoff_spin.setPrefix("Cutoff: ")
        self.cutoff_spin.setSuffix(" kHz")
        controls.addWidget(self.cutoff_spin)
        self.hp_cutoff_spin = QDoubleSpinBox()
        self.hp_cutoff_spin.setRange(0.01, max(0.01, self.fs / 2000 - 0.01))
        self.hp_cutoff_spin.setSingleStep(0.1)
        self.hp_cutoff_spin.setValue(0.5)
        self.hp_cutoff_spin.setPrefix("High: ")
        self.hp_cutoff_spin.setSuffix(" kHz")
        self.hp_cutoff_spin.setVisible(False)
        controls.addWidget(self.hp_cutoff_spin)
        self.lp_cutoff_spin = QDoubleSpinBox()
        self.lp_cutoff_spin.setRange(0.01, max(0.01, self.fs / 2000 - 0.01))
        self.lp_cutoff_spin.setSingleStep(0.1)
        self.lp_cutoff_spin.setValue(2.0)
        self.lp_cutoff_spin.setPrefix("Low: ")
        self.lp_cutoff_spin.setSuffix(" kHz")
        self.lp_cutoff_spin.setVisible(False)
        controls.addWidget(self.lp_cutoff_spin)
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(1, 101)
        self.smooth_spin.setValue(5)
        self.smooth_spin.setPrefix("Window: ")
        self.smooth_spin.setSuffix(" samples")
        self.smooth_spin.setVisible(False)
        controls.addWidget(self.smooth_spin)
        self.apply_btn = QPushButton("Apply")
        controls.addWidget(self.apply_btn)
        self.save_btn = QPushButton("Save Waveform (PNG, 900 DPI)")
        self.save_btn.clicked.connect(self.save_fig)
        controls.addWidget(self.save_btn)
        self.clear_picks_btn = QPushButton("Clear Manual Picks")
        self.clear_picks_btn.clicked.connect(self.clear_manual_picks)
        controls.addWidget(self.clear_picks_btn)
        # Restore Show Manual Picks and Export/Reflection Buttons if desired
        self.show_all_horizons_btn = QPushButton("Show All Horizons on Cosine Phase")
        self.show_all_horizons_btn.clicked.connect(self.handle_show_all_horizons)
        controls.addWidget(self.show_all_horizons_btn)
        self.export_picks_btn = QPushButton("Export Manual Picks (.xlsx)")
        self.export_picks_btn.clicked.connect(self.handle_export_manual_picks)
        controls.addWidget(self.export_picks_btn)
        self.reflection_btn = QPushButton("Reflection Coefficient")
        self.reflection_btn.clicked.connect(self.handle_reflection_coeff)
        controls.addWidget(self.reflection_btn)
        layout.addLayout(controls)
        # Matplotlib figure and toolbar for waveform visualization
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # --- Connect filter controls and apply logic ---
        self.filter_combo.currentIndexChanged.connect(self.update_controls_visibility)
        self.filter_combo.currentIndexChanged.connect(self.apply_filter_and_plot)
        self.cutoff_spin.valueChanged.connect(self.apply_filter_and_plot)
        self.hp_cutoff_spin.valueChanged.connect(self.apply_filter_and_plot)
        self.lp_cutoff_spin.valueChanged.connect(self.apply_filter_and_plot)
        self.smooth_spin.valueChanged.connect(self.apply_filter_and_plot)
        self.apply_btn.clicked.connect(self.apply_filter_and_plot)
        self.update_controls_visibility()
        # Connect picking
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.plot_waveform(self.filtered_trace, label="Raw Signal", color='gray')

    def handle_show_all_horizons(self):
        mainwin = self.parent()
        if not hasattr(mainwin, "phase") or mainwin.phase is None:
            QMessageBox.warning(self, "No Phase Data", "No phase data available.")
            return
        phase = mainwin.phase
        horizon_picks = getattr(mainwin, "horizon_picks", [])
        manual_picks_dict = getattr(mainwin, "manual_picks_dict", {})
        dlg = CosinePhaseHorizonsDialog(
            phase,
            horizon_picks,
            manual_picks_dict,
            parent=self
        )
        dlg.exec_()

    def update_controls_visibility(self):
        """
        Show/hide filter parameter controls depending on selected filter type.
        """
        ftype = self.filter_combo.currentText()
        self.cutoff_spin.setVisible(ftype in ["Low-pass", "High-pass"])
        self.hp_cutoff_spin.setVisible(ftype == "Band-pass")
        self.lp_cutoff_spin.setVisible(ftype == "Band-pass")
        self.smooth_spin.setVisible(ftype == "Smoothing")

    def apply_filter_and_plot(self):
        """
        Apply the selected filter to the trace and update the plot.
        """
        ftype = self.filter_combo.currentText()
        y = self.trace
        try:
            if ftype == "None":
                self.filtered_trace = y.copy()
                self.plot_waveform(self.filtered_trace, label="Raw Signal", color='gray')
            elif ftype == "Low-pass":
                cutoff = self.cutoff_spin.value() * 1000
                self.filtered_trace = butter_filter(y, self.fs, 'low', cutoff)
                self.plot_waveform(self.filtered_trace, label=f"Low-pass {self.cutoff_spin.value():.2f}kHz", color='b')
            elif ftype == "High-pass":
                cutoff = self.cutoff_spin.value() * 1000
                self.filtered_trace = butter_filter(y, self.fs, 'high', cutoff)
                self.plot_waveform(self.filtered_trace, label=f"High-pass {self.cutoff_spin.value():.2f}kHz", color='g')
            elif ftype == "Band-pass":
                hp = self.hp_cutoff_spin.value() * 1000
                lp = self.lp_cutoff_spin.value() * 1000
                if hp >= lp:
                    QMessageBox.warning(self, "Invalid band", "High cutoff must be lower than low cutoff")
                    return
                self.filtered_trace = butter_filter(y, self.fs, 'band', band=[hp, lp])
                self.plot_waveform(self.filtered_trace, label=f"Band {self.hp_cutoff_spin.value():.2f}-{self.lp_cutoff_spin.value():.2f}kHz", color='m')
            elif ftype == "Smoothing":
                window = self.smooth_spin.value()
                self.filtered_trace = np.convolve(y, np.ones(window)/window, mode='same')
                self.plot_waveform(self.filtered_trace, label=f"Smoothing {window}", color='r')
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Error applying filter: {e}")

    def plot_waveform(self, y, label="Raw Signal", color='gray'):
        x_depth = twt_to_depth(self.sample_depths)
        self.ax.clear()
        if label != "Raw Signal":
            self.ax.plot(x_depth, self.trace, color='gray', lw=0.5, label="Raw Signal", zorder=1)
        self.ax.plot(x_depth, y, color=color, lw=0.5, label=label, zorder=2)
        # Plot min/max cutoff lines if indices are set
        min_sample_index = getattr(self.parent(), 'min_sample_spin', None)
        max_sample_index = getattr(self.parent(), 'max_sample_spin', None)
        min_label_plotted = False
        max_label_plotted = False
        if min_sample_index is not None and hasattr(min_sample_index, "value"):
            min_idx = min_sample_index.value()
            if 0 <= min_idx < len(self.sample_depths):
                cutoff_depth = twt_to_depth(self.sample_depths[min_idx])
                self.ax.axvline(cutoff_depth, color='red', linestyle='--', lw=1.5, label="Picking Cutoff")
                min_label_plotted = True
        if self.max_sample_index is not None and 0 <= self.max_sample_index < len(self.sample_depths):
            cutoff_depth = twt_to_depth(self.sample_depths[self.max_sample_index])
            label = "Picking Cutoff" if not min_label_plotted or self.max_sample_index != min_sample_index.value() else ""
            self.ax.axvline(cutoff_depth, color='red', linestyle='--', lw=1.5, label=label)
            max_label_plotted = True
        # Plot manual picks
        if len(self.manual_picks) > 0:
            for i, idx in enumerate(self.manual_picks):
                self.ax.axvline(x_depth[idx], color='g', linestyle='-.', lw=1.5, alpha=1,
                                label="Manual Pick" if i == 0 else "")
        self.ax.set_xlabel("Depth (m)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title(f"Waveform at Selected Trace {self.trace_idx}")
        handles, labels = self.ax.get_legend_handles_labels()
        uniq = list(dict(zip(labels, handles)).items())
        if uniq:
            self.ax.legend([h for l, h in uniq], [l for l, h in uniq])
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.fig.tight_layout()
        self.canvas.draw()

    def on_canvas_click(self, event):
        """
        Handle manual pick (left-click) on the waveform plot.
        """
        if event.inaxes != self.ax or event.button != 1:
            return
        x_depth = twt_to_depth(self.sample_depths)
        click_depth = event.xdata
        if click_depth is None:
            return
        idx = np.argmin(np.abs(x_depth - click_depth))
        if self.max_sample_index is not None and idx > self.max_sample_index:
            QMessageBox.warning(self, "Invalid Pick", "Cannot pick below the cutoff depth.")
            return
        if idx not in self.manual_picks:
            self.manual_picks.append(idx)
            self.plot_waveform(self.filtered_trace,
                               label=self.filter_combo.currentText() if self.filter_combo.currentText() != "None" else "Raw Signal",
                               color={'None':'gray','Low-pass':'b','High-pass':'g','Band-pass':'m','Smoothing':'r'}.get(self.filter_combo.currentText(), 'gray'))
            self.manual_picks_dict[self.trace_idx] = list(self.manual_picks)

    def clear_manual_picks(self):
        """
        Clear all manual picks for this trace and update the plot.
        """
        self.manual_picks = []
        self.plot_waveform(self.filtered_trace,
                           label=self.filter_combo.currentText() if self.filter_combo.currentText() != "None" else "Raw Signal",
                           color={'None':'gray','Low-pass':'b','High-pass':'g','Band-pass':'m','Smoothing':'r'}.get(self.filter_combo.currentText(), 'gray'))
        if self.trace_idx in self.manual_picks_dict:
            del self.manual_picks_dict[self.trace_idx]

    def save_fig(self):
        """
        Export the current waveform plot as a PNG image.
        """
        fname, _ = QFileDialog.getSaveFileName(self, "Save Waveform Figure", "", "PNG Image (*.png)")
        if fname:
            self.fig.savefig(fname, dpi=900, bbox_inches='tight')

    def handle_export_manual_picks(self):
        """
        Export manual picks for this trace to an Excel file.
        """
        if self.trace_idx not in self.manual_picks_dict or not self.manual_picks_dict[self.trace_idx]:
            QMessageBox.information(self, "No Picks", "No manual picks to export for this trace.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Export Manual Picks", "", "Excel Files (*.xlsx)")
        if not fname:
            return
        rows = []
        for idx in self.manual_picks_dict[self.trace_idx]:
            depth = twt_to_depth(self.sample_depths[idx])
            rows.append({'Trace Index': self.trace_idx, 'Sample Index': idx, 'Depth (m)': depth})
        df = pd.DataFrame(rows)
        df.to_excel(fname, index=False)
        QMessageBox.information(self, "Export Complete", f"Manual picks exported to {fname}")

    def handle_show_manual_picks(self):
        """
        Show manual and automatic picks for this trace on a new cosine phase image window.
        """
        mainwin = self.parent()
        if not hasattr(mainwin, "phase") or mainwin.phase is None:
            QMessageBox.warning(self, "No Phase Data", "No phase data available.")
            return
        phase = mainwin.phase
        # Get automatic picks for this trace
        auto_picks = []
        if hasattr(mainwin, "horizon_picks") and mainwin.horizon_picks:
            auto_picks = [s for t, s in mainwin.horizon_picks if t == self.trace_idx]
        manual_picks = self.manual_picks_dict.get(self.trace_idx, [])
        dlg = TracePicksDialog(
            self.trace_idx,
            phase,
            manual_picks,
            auto_picks,
            self.sample_depths,
            parent=self
        )
        dlg.exec_()

    def handle_reflection_coeff(self):
        """
        Show the Reflection Coefficient dialog for current manual picks.
        """
        if self.trace_idx not in self.manual_picks_dict or not self.manual_picks_dict[self.trace_idx]:
            QMessageBox.information(self, "No Picks", "No manual picks to analyze for this trace.")
            return
        picks = self.manual_picks_dict[self.trace_idx]
        dlg = ReflectionCoeffDialog(self.trace, self.sample_depths, picks, parent=self)
        dlg.exec_()

# --------------------------- MAIN GUI APPLICATION ---------------------------
class ModernHorizonPickerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SBP Horizon Picker GUI")
        self.setGeometry(50, 50, 1400, 900)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.seismic_data = None
        self.sample_depths = None
        self.horizons = []
        self.local_orientations = []
        self.envelope_smoothed = None
        self.phase = None
        self.worker = None
        self.cpu_count = multiprocessing.cpu_count()
        self.manual_picks_dict = {}
        self.initUI()
        self.canvas.mpl_connect('button_press_event', self.on_seed_click)

    def initUI(self):
        main_layout = QHBoxLayout(self.central_widget)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setAlignment(Qt.AlignTop)
        self.info_label = QLabel()
        controls_layout.addWidget(self.info_label)
        self.btn_load = QPushButton("Load SEG-Y")
        self.btn_load.clicked.connect(self.load_segy)
        controls_layout.addWidget(self.btn_load)
        controls_layout.addWidget(QLabel("Envelope Threshold Ratio"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(1)
        controls_layout.addWidget(self.thresh_spin)
        controls_layout.addWidget(QLabel("Smoothing Sigma"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 20.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(8)
        controls_layout.addWidget(self.sigma_spin)
        controls_layout.addWidget(QLabel("Min Distance"))
        self.mindist_spin = QSpinBox()
        self.mindist_spin.setRange(1, 50)
        self.mindist_spin.setValue(10)
        controls_layout.addWidget(self.mindist_spin)
        controls_layout.addWidget(QLabel("Max Horizons/Trace"))
        self.maxh_spin = QSpinBox()
        self.maxh_spin.setRange(1, 10)
        self.maxh_spin.setValue(2)
        controls_layout.addWidget(self.maxh_spin)
        controls_layout.addWidget(QLabel("Ignore Top Samples"))
        self.min_sample_spin = QSpinBox()
        self.min_sample_spin.setRange(0, 100)
        self.min_sample_spin.setValue(10)
        controls_layout.addWidget(self.min_sample_spin)
        controls_layout.addWidget(QLabel("Min Sample Index"))
        self.min_sample_spin = QSpinBox()
        self.min_sample_spin.setRange(0, 10000)
        self.min_sample_spin.setValue(0)
        self.min_sample_spin.setToolTip("Min Sample Index")
        controls_layout.addWidget(self.min_sample_spin)
        controls_layout.addWidget(QLabel("Max Sample Index"))
        self.max_sample_spin = QSpinBox()
        self.max_sample_spin.setRange(1, 10000)
        self.max_sample_spin.setValue(500)
        controls_layout.addWidget(self.max_sample_spin)
        self.btn_pick = QPushButton("Pick Horizons")
        self.btn_pick.clicked.connect(self.pick_and_plot)
        controls_layout.addWidget(self.btn_pick)
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        controls_layout.addWidget(self.progress)
        controls_layout.addSpacing(20)
        self.btn_show_dip_az = QPushButton("Show Dip/Azimuth/Rose")
        self.btn_show_dip_az.clicked.connect(self.show_dip_azimuth_dialog)
        controls_layout.addWidget(self.btn_show_dip_az)
        self.btn_waveform = QPushButton("Show Waveform")
        self.btn_waveform.clicked.connect(self.show_waveform_dialog)  
        controls_layout.addWidget(self.btn_waveform)
        self.btn_export_auto_xlsx = QPushButton("Export Automatic Picks to .xlsx")
        self.btn_export_auto_xlsx.clicked.connect(self.export_automatic_picks_to_xlsx)
        controls_layout.addWidget(self.btn_export_auto_xlsx)
        controls_layout.addStretch()
        main_panel = QWidget()
        panel_layout = QVBoxLayout(main_panel)
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 16))
        self.canvas = FigureCanvas(self.fig)
        panel_layout.addWidget(self.canvas)
        self.save_main_btn = QPushButton("Save Displayed Graphs (PNG, 900 DPI)")
        panel_layout.addWidget(self.save_main_btn)
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(main_panel)
        main_layout.setStretch(1, 1)

    def load_segy(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open SEG-Y File", "",
            "SEG-Y Files (*.sgy *.segy);;All Files (*)", options=options)
        if file_name:
            try:
                with segyio.open(file_name, "r", ignore_geometry=True) as f:
                    num_traces = f.tracecount
                    num_samples = f.samples.size
                    data = np.empty((num_traces, num_samples), dtype=np.float32)
                    for i in range(num_traces):
                        data[i, :] = f.trace[i]
                    depths = np.array(f.samples)
                self.seismic_data = data
                dt_ms = float(depths[1] - depths[0]) if len(depths) > 1 else 1.0
                self.info_label.setText(
                    f"Loaded SBP data shape: {self.seismic_data.shape}, sample interval: {dt_ms:.3f} ms")
                self.sample_depths = depths
                self.horizons = []
                self.local_orientations = []
                self.envelope_smoothed = None
                self.phase = None
                self.plot_all()
                self.max_sample_spin.setMaximum(num_samples - 1)
                self.max_sample_spin.setValue(num_samples - 1)
                self.manual_picks_dict = {}
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load SEG-Y file:\n{e}")

    def pick_and_plot(self):
        if self.seismic_data is None:
            QMessageBox.warning(self, "No Data", "Please load a SEG-Y file first.")
            return
        params = {
            'env_thresh_ratio': self.thresh_spin.value(),
            'sigma': self.sigma_spin.value(),
            'max_horizons': self.maxh_spin.value(),
            'min_dist': self.mindist_spin.value(),
            'min_sample': self.min_sample_spin.value(),
            'max_sample': self.max_sample_spin.value(),
        }
        self.progress.setValue(0)
        self.btn_pick.setEnabled(False)
        self.worker = HorizonWorker(self.seismic_data, self.sample_depths, params)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_worker_result)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_result(self, result):
        self.btn_pick.setEnabled(True)
        self.progress.setValue(100)
        self.horizons = result['horizons']
        self.local_orientations = result['orientations']
        self.envelope_smoothed = result['envelope']
        self.phase = result['phase']
        self.plot_all()

    def _on_worker_error(self, msg):
        self.btn_pick.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error during picking:\n{msg}")

    def plot_all(self):
        print("plot_all: horizons found:", len(self.horizons))
        if self.horizons:
            print([sum(s is not None for s in h) for h in self.horizons])
        self.axs[0].clear()
        if self.seismic_data is not None:
            self.axs[0].imshow(self.seismic_data.T, cmap='gray', aspect='auto')
            self.axs[0].set_title('Raw SBP Profile')
            self.axs[0].set_ylabel('Sample Index (or TWT)')

        self.axs[1].clear()
        if self.phase is not None:
            # Show cosine phase using extent to align depth axis
            depth_min = twt_to_depth(self.sample_depths[0])
            depth_max = twt_to_depth(self.sample_depths[-1])
            img = self.axs[1].imshow(
                np.cos(self.phase).T,
                cmap='gray',
                aspect='auto',
                extent=[0, self.seismic_data.shape[0], depth_max, depth_min]
            )
            # y-ticks in depth
            if self.sample_depths is not None:
                num_ticks = 8
                sample_indices = np.linspace(0, len(self.sample_depths) - 1, num_ticks, dtype=int)
                depth_labels = [f"{twt_to_depth(self.sample_depths[i]):.1f}" for i in sample_indices]
                depth_tickvals = [twt_to_depth(self.sample_depths[i]) for i in sample_indices]
                self.axs[1].set_yticks(depth_tickvals)
                self.axs[1].set_yticklabels(depth_labels)
                self.axs[1].set_ylabel('Depth (m)')
            else:
                self.axs[1].set_ylabel('Sample Index')
            # Plot horizons in depth space
            if self.horizons:
                colors = ['r', 'g', 'b', 'm', 'c']
                for idx, horizon in enumerate(self.horizons):
                    pick_i = [i for i, s in enumerate(horizon) if s is not None]
                    pick_j = [s for s in horizon if s is not None]
                    pick_depth = [twt_to_depth(self.sample_depths[s]) for s in pick_j]
                    color = colors[idx % len(colors)]
                    self.axs[1].plot(pick_i, pick_depth, '.', markersize=2, color=color, label=f"Horizon {idx + 1}")
                self.axs[1].legend(fontsize=6)
            self.axs[1].set_xlabel('Trace Index')
            self.axs[1].set_title('Cosine Phase Image with Horizons')
        else:
            self.axs[1].set_title('Cosine Phase Image')
            self.axs[1].set_ylabel('Sample Index (or TWT)')

        self.axs[2].clear()
        if self.envelope_smoothed is not None:
            gray_env = envelope_to_grayscale(self.envelope_smoothed)
            depth_min = twt_to_depth(self.sample_depths[0])
            depth_max = twt_to_depth(self.sample_depths[-1])
            self.axs[2].imshow(
                gray_env.T,
                cmap='gray',
                aspect='auto',
                extent=[0, self.seismic_data.shape[0], depth_max, depth_min]
            )
            if self.sample_depths is not None:
                num_ticks = 8
                sample_indices = np.linspace(0, len(self.sample_depths) - 1, num_ticks, dtype=int)
                depth_labels = [f"{twt_to_depth(self.sample_depths[i]):.1f}" for i in sample_indices]
                depth_tickvals = [twt_to_depth(self.sample_depths[i]) for i in sample_indices]
                self.axs[2].set_yticks(depth_tickvals)
                self.axs[2].set_yticklabels(depth_labels)
                self.axs[2].set_ylabel('Depth (m)')
            else:
                self.axs[2].set_ylabel('Sample Index')
            if self.horizons:
                colors = ['r', 'g', 'b', 'm', 'c']
                for idx, horizon in enumerate(self.horizons):
                    pick_i = [i for i, s in enumerate(horizon) if s is not None]
                    pick_j = [s for s in horizon if s is not None]
                    pick_depth = [twt_to_depth(self.sample_depths[s]) for s in pick_j]
                    color = colors[idx % len(colors)]
                    self.axs[2].plot(pick_i, pick_depth, '.', markersize=2, color=color, label=f"Horizon {idx + 1}")
                self.axs[2].legend(fontsize=6)
            self.axs[2].set_xlabel('Trace index')
            self.axs[2].set_title('Envelope Image (Haralick Transform)')
        else:
            self.axs[2].set_title('Envelope Image (Haralick Transform)')
            self.axs[2].set_xlabel('Trace index')
            self.axs[2].set_ylabel('Sample Index (or TWT)')
        self.fig.tight_layout()
        self.canvas.draw()

    def show_waveform_dialog(self):
        if self.seismic_data is None or self.sample_depths is None:
            QMessageBox.warning(self, "No Data", "Please load a SEG-Y file first.")
            return
        num_traces = self.seismic_data.shape[0]
        # Prompt the user to select the trace index within valid range
        trace_idx, ok = QInputDialog.getInt(
            self,
            "Select Trace",
            f"Select trace index (0 - {num_traces - 1}):",
            value=0,
            min=0,
            max=num_traces - 1
        )
        if not ok:
            return
        trace = self.seismic_data[trace_idx]
        max_sample_index = self.seismic_data.shape[1] - 1
        dlg = WaveformDialog(
            trace,
            self.sample_depths,
            trace_idx,
            parent=self,
            manual_picks_dict=self.manual_picks_dict,
            max_sample_index=max_sample_index
        )
        dlg.exec_()

    def show_dip_azimuth_dialog(self):
        if not (self.horizons and self.local_orientations):
            QMessageBox.information(self, "No Data", "No horizon data to show.")
            return
        dlg = DipAzimuthDialog(self.horizons, self.local_orientations, self)
        dlg.exec_()

    def export_automatic_picks_to_xlsx(self):
        if not hasattr(self, "horizons") or not self.horizons:
            QMessageBox.information(self, "No Picks", "No automatic picks to export.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Export Automatic Picks", "", "Excel Files (*.xlsx)")
        if not fname:
            return
        rows = []
        for h_idx, horizon in enumerate(self.horizons):
            for trace_idx, sample_idx in enumerate(horizon):
                if sample_idx is not None:
                    rows.append({
                        'Horizon': h_idx + 1,
                        'Trace Index': trace_idx,
                        'Sample Index': sample_idx,
                        'Depth (m)': twt_to_depth(self.sample_depths[sample_idx])
                    })
        df = pd.DataFrame(rows)
        df.to_excel(fname, index=False)
        QMessageBox.information(self, "Export Complete", f"Automatic picks exported to {fname}")

    def on_seed_click(self, event):
        pass  

# ---------------------------- MAIN ENTRY POINT ------------------------------
def main():
    app = QApplication(sys.argv)
    win = ModernHorizonPickerGUI()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
